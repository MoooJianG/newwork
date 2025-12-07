"""
2D Gaussian Splatting module for continuous-scale super-resolution.

This module uses Deep Gaussian Prior with predefined covariance dictionary
and gsplat CUDA acceleration for efficient rendering.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

try:
    from gsplat.project_gaussians_2d import project_gaussians_2d
    from gsplat.rasterize_sum import rasterize_gaussians_sum
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False


def make_coord(shape, flatten=True):
    """Generate normalized coordinate grid in range [-1, 1]."""
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class GaussianSplattingV2(nn.Module):
    """
    2D Gaussian Splatting module with Deep Gaussian Prior.
    
    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output channels (default: 3 for RGB).
        hidden_dim: Hidden dimension for MLPs.
        dict_size: Size of covariance dictionary (default: 729 = 9^3).
        use_gsplat: Whether to use gsplat CUDA acceleration.
        upsample_factor: Factor to increase Gaussian point density.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels=3,
        hidden_dim=256,
        dict_size=729,
        use_gsplat=True,
        upsample_factor=2  # 高斯点密度倍数 (ContinuousSR用2)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.use_gsplat = use_gsplat and GSPLAT_AVAILABLE
        self.upsample_factor = upsample_factor
        self.BLOCK_H, self.BLOCK_W = 16, 16
        
        # Feature extraction with optional upsampling
        if upsample_factor > 1:
            self.feat_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim * (upsample_factor ** 2), 3, padding=1),
                nn.PixelShuffle(upsample_factor),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
        else:
            self.feat_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            )
        
        # Color prediction MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, out_channels),
        )
        
        # Position offset prediction MLP
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
        
        # Covariance dictionary selector
        self.cov_selector = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, 512, 1),
        )
        
        # Initialize covariance dictionary
        self._init_covariance_dict(dict_size)
        
        # Dictionary embedding MLP
        self.dict_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        
        # Opacity prediction MLP
        self.opacity_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.register_buffer('background', torch.ones(3))
        
    def _init_covariance_dict(self, dict_size):
        """Initialize predefined covariance dictionary in Cholesky form."""
        grid_size = int(round(dict_size ** (1/3)))
        
        cho1 = torch.linspace(0, 2.5, grid_size)
        cho2 = torch.linspace(-1.0, 1.5, grid_size)
        cho3 = torch.linspace(0, 2.5, grid_size)
        
        gau_dict = torch.tensor(list(product(cho1, cho2, cho3)), dtype=torch.float32)
        gau_dict = torch.cat([gau_dict, torch.zeros(1, 3)], dim=0)
        
        self.register_buffer('gau_dict', gau_dict)
        self.dict_size = gau_dict.shape[0]
        
    def _get_weighted_covariance(self, feat):
        """Get weighted covariance parameters via dictionary lookup."""
        B, C, H, W = feat.shape
        
        selector_feat = self.cov_selector(feat)
        selector_feat = selector_feat.permute(0, 2, 3, 1).reshape(B * H * W, 512)
        
        dict_embed = self.dict_embed(self.gau_dict)
        similarity = torch.matmul(selector_feat, dict_embed.T)
        weights = F.softmax(similarity / math.sqrt(512), dim=-1)
        
        cov_params = torch.matmul(weights, self.gau_dict)
        cov_params = cov_params.reshape(B, H * W, 3)
        
        return cov_params
    
    def _predict_colors_and_offsets(self, feat):
        """Predict per-pixel colors, offsets and opacities."""
        B, C, H, W = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        colors = self.color_mlp(feat_flat).reshape(B, H * W, self.out_channels)
        offsets = self.offset_mlp(feat_flat).reshape(B, H * W, 2)
        opacities = self.opacity_mlp(feat_flat).reshape(B, H * W, 1)
        
        return colors, offsets, opacities

    def _render_gsplat(self, coords, cov_params, colors, opacities, H_out, W_out, scale, lr_h, lr_w):
        """Render using gsplat CUDA library."""
        B, N, _ = coords.shape
        
        tile_bounds = (
            (W_out + self.BLOCK_W - 1) // self.BLOCK_W,
            (H_out + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        
        outputs = []
        for i in range(B):
            coord_i = coords[i].clone()
            coord_i[:, 0] = coord_i[:, 0] - 1.0 / W_out
            coord_i[:, 1] = coord_i[:, 1] - 1.0 / H_out
            
            cov_i = cov_params[i].clone() / 4
            color_i = colors[i]
            opacity_i = opacities[i]
            
            cov_i[:, 0] *= scale
            cov_i[:, 1] *= scale
            cov_i[:, 2] *= scale
            
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                coord_i, cov_i, H_out, W_out, tile_bounds
            )
            
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, opacity_i,
                H_out, W_out, self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False
            )
            
            outputs.append(out_img.permute(2, 0, 1).unsqueeze(0))
            
        return torch.cat(outputs, dim=0)
    
    def _render_pytorch(self, coords, cov_params, colors, opacities, H_out, W_out, scale):
        """Pure PyTorch rendering fallback."""
        B, N, _ = coords.shape
        device = coords.device
        
        y_coords = torch.linspace(-1, 1, H_out, device=device)
        x_coords = torch.linspace(-1, 1, W_out, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixel_coords = torch.stack([yy, xx], dim=-1)
        
        cho1 = cov_params[..., 0:1] * scale + 1e-4
        cho2 = cov_params[..., 1:2] * scale
        cho3 = cov_params[..., 2:3] * scale + 1e-4
        
        sigma_xx = cho1 ** 2
        sigma_xy = cho1 * cho2
        sigma_yy = cho2 ** 2 + cho3 ** 2
        
        det = sigma_xx * sigma_yy - sigma_xy ** 2 + 1e-6
        inv_xx = sigma_yy / det
        inv_xy = -sigma_xy / det
        inv_yy = sigma_xx / det
        
        coords_expanded = coords.unsqueeze(2).unsqueeze(2)
        pixel_expanded = pixel_coords.unsqueeze(0).unsqueeze(0)
        
        diff = pixel_expanded - coords_expanded
        dx = diff[..., 0]
        dy = diff[..., 1]
        
        inv_xx = inv_xx.unsqueeze(-1)
        inv_xy = inv_xy.unsqueeze(-1)
        inv_yy = inv_yy.unsqueeze(-1)
        
        mahal = dx ** 2 * inv_xx + 2 * dx * dy * inv_xy + dy ** 2 * inv_yy
        gaussian_weight = torch.exp(-0.5 * mahal)
        
        opacity_expanded = opacities.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        weighted_gaussian = gaussian_weight * opacity_expanded
        
        colors_expanded = colors.unsqueeze(-1).unsqueeze(-1)
        weighted_colors = weighted_gaussian.unsqueeze(2) * colors_expanded
        
        output = weighted_colors.sum(dim=1)
        weight_sum = weighted_gaussian.sum(dim=1, keepdim=True)
        
        output = output / (weight_sum + 1e-6)
        alpha = 1 - torch.exp(-weight_sum)
        output = output * alpha + self.background.view(1, 3, 1, 1) * (1 - alpha)
        
        return output

    def forward(self, feat, lr, out_size):
        """Forward pass."""
        B, C, H_in, W_in = feat.shape
        H_out, W_out = out_size
        device = feat.device
        
        feat = self.feat_conv(feat)
        _, _, H, W = feat.shape
        scale = H_out / H
        
        colors, offsets, opacities = self._predict_colors_and_offsets(feat)
        cov_params = self._get_weighted_covariance(feat)
        
        base_coords = make_coord((H, W), flatten=True).to(device)
        base_coords = base_coords.unsqueeze(0).expand(B, -1, -1)
        
        pixel_size = 2.0 / H
        coords = base_coords + offsets * pixel_size
        
        if self.use_gsplat:
            out = self._render_gsplat(coords, cov_params, colors, opacities, H_out, W_out, scale, H, W)
        else:
            out = self._render_pytorch(coords, cov_params, colors, opacities, H_out, W_out, scale)
        
        lr_up = F.interpolate(lr, out_size, mode='bicubic', align_corners=False)
        out = out + lr_up
        
        return out
    
    def get_last_layer(self):
        """Return last layer weight for GAN training."""
        return self.color_mlp[-1].weight


class GaussianSplattingDecoderV2(nn.Module):
    """Decoder wrapper compatible with LDCSR interface."""
    
    def __init__(self, in_channels, out_channels=3, hidden_dim=256, 
                 dict_size=729, use_gsplat=True, **kwargs):
        super().__init__()
        self.gaussian_splatting = GaussianSplattingV2(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            dict_size=dict_size,
            use_gsplat=use_gsplat
        )
        
    def forward(self, feat, lr, out_size):
        return self.gaussian_splatting(feat, lr, out_size)
    
    def get_last_layer(self):
        return self.gaussian_splatting.get_last_layer()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, gsplat: {GSPLAT_AVAILABLE}")
    
    model = GaussianSplattingV2(64, 3, 256, 729, True).to(device)
    print(f"Using gsplat: {model.use_gsplat}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    feat = torch.randn(2, 64, 32, 32).to(device)
    lr = torch.randn(2, 3, 32, 32).to(device)
    
    for s in [2.0, 3.0, 4.0]:
        out = model(feat, lr, (int(32 * s), int(32 * s)))
        print(f"  {s}x: {feat.shape} -> {out.shape}")
