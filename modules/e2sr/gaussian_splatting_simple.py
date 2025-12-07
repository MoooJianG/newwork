"""
Simplified 2D Gaussian Splatting module for continuous-scale super-resolution.

Uses predefined covariance dictionary and gsplat CUDA rendering.
"""

import math
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
        r = 1.0 / n
        seq = -1 + r + 2 * r * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class GaussianSplattingSimple(nn.Module):
    """
    Simplified 2D Gaussian Splatting module.
    
    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output channels (default: 3 for RGB).
        hidden_dim: Hidden dimension for MLPs.
    """
    
    def __init__(self, in_channels, out_channels=3, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.BLOCK_H, self.BLOCK_W = 16, 16
        
        # Feature processing with PixelUnshuffle
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.ps = nn.PixelUnshuffle(2)
        
        # Covariance prediction
        self.conv_cov = nn.Conv2d(hidden_dim * 4, 512, 3, padding=1)
        
        # Color prediction MLP
        self.mlp_color = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels),
        )
        
        # Offset prediction MLP
        self.mlp_offset = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh(),
        )
        
        # Dictionary embedding MLP
        self.mlp_dict = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        
        self._init_gaussian_dict()
        self.register_buffer('background', torch.ones(3))
        
    def _init_gaussian_dict(self):
        """Initialize predefined covariance dictionary in Cholesky form."""
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])
        
        gau_dict = torch.tensor(list(product(cho1, cho2, cho3)), dtype=torch.float32)
        gau_dict = torch.cat([gau_dict, torch.zeros(1, 3)], dim=0)
        
        self.register_buffer('gau_dict', gau_dict)
        
    def forward(self, feat, lr, out_size):
        """Forward pass."""
        B, C, H, W = feat.shape
        H_out, W_out = out_size
        device = feat.device
        
        scale_h = H_out / H
        scale_w = W_out / W
        
        feat = self.conv1(feat)
        feat = self.act(feat)
        feat_ps = self.ps(feat)
        
        h, w = feat_ps.shape[2], feat_ps.shape[3]
        num_points = h * w
        
        feat_flat = feat_ps.permute(0, 2, 3, 1).reshape(B * num_points, -1)
        colors = self.mlp_color(feat_flat).reshape(B, num_points, self.out_channels)
        offsets = self.mlp_offset(feat_flat).reshape(B, num_points, 2)
        
        cov_feat = self.conv_cov(feat_ps)
        cov_feat = cov_feat.permute(0, 2, 3, 1).reshape(B * num_points, 512)
        
        dict_embed = self.mlp_dict(self.gau_dict)
        similarity = cov_feat @ dict_embed.T
        weights = F.softmax(similarity, dim=-1)
        cov_params = weights @ self.gau_dict
        cov_params = cov_params.reshape(B, num_points, 3)
        
        base_coords = make_coord((h, w), flatten=True).to(device)
        base_coords = base_coords.unsqueeze(0).expand(B, -1, -1)
        coords = base_coords + offsets * (2.0 / h)
        
        if GSPLAT_AVAILABLE:
            out = self._render_gsplat(coords, cov_params, colors, H_out, W_out, scale_h, scale_w)
        else:
            out = self._render_simple(coords, cov_params, colors, H_out, W_out, scale_h)
        
        lr_up = F.interpolate(lr, out_size, mode='bicubic', align_corners=False)
        out = out + lr_up
        
        return out

    def _render_gsplat(self, coords, cov_params, colors, H_out, W_out, scale_h, scale_w):
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
            cov_i = cov_params[i].clone() / 4
            color_i = colors[i]
            opacity_i = torch.ones(N, 1, device=coords.device)
            
            cov_i[:, 0] *= scale_w
            cov_i[:, 1] *= scale_h
            cov_i[:, 2] *= scale_h
            
            coord_i[:, 0] -= 1.0 / W_out
            coord_i[:, 1] -= 1.0 / H_out
            
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
    
    def _render_simple(self, coords, cov_params, colors, H_out, W_out, scale):
        """Pure PyTorch rendering fallback."""
        B, N, _ = coords.shape
        device = coords.device
        
        y = torch.linspace(-1, 1, H_out, device=device)
        x = torch.linspace(-1, 1, W_out, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pixels = torch.stack([yy, xx], dim=-1)
        
        cho1 = (cov_params[..., 0:1] * scale + 1e-4).unsqueeze(-1).unsqueeze(-1)
        cho2 = (cov_params[..., 1:2] * scale).unsqueeze(-1).unsqueeze(-1)
        cho3 = (cov_params[..., 2:3] * scale + 1e-4).unsqueeze(-1).unsqueeze(-1)
        
        sigma_xx = cho1 ** 2
        sigma_xy = cho1 * cho2
        sigma_yy = cho2 ** 2 + cho3 ** 2
        det = sigma_xx * sigma_yy - sigma_xy ** 2 + 1e-6
        
        coords_exp = coords.unsqueeze(2).unsqueeze(2)
        pixels_exp = pixels.unsqueeze(0).unsqueeze(0)
        diff = pixels_exp - coords_exp
        
        mahal = (diff[..., 0:1] ** 2 * sigma_yy / det + 
                 diff[..., 1:2] ** 2 * sigma_xx / det - 
                 2 * diff[..., 0:1] * diff[..., 1:2] * sigma_xy / det)
        
        weights = torch.exp(-0.5 * mahal.squeeze(-1))
        
        colors_exp = colors.unsqueeze(-1).unsqueeze(-1)
        out = (weights.unsqueeze(2) * colors_exp).sum(dim=1)
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        out = out / weight_sum
        
        return out
    
    def get_last_layer(self):
        """Return last layer weight for GAN training."""
        return self.mlp_color[-1].weight


class GaussianSplattingDecoderSimple(nn.Module):
    """Decoder wrapper compatible with LDCSR interface."""
    
    def __init__(self, in_channels, out_channels=3, hidden_dim=256, **kwargs):
        super().__init__()
        self.gs = GaussianSplattingSimple(in_channels, out_channels, hidden_dim)
        
    def forward(self, feat, lr, out_size):
        return self.gs(feat, lr, out_size)
    
    def get_last_layer(self):
        return self.gs.get_last_layer()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, gsplat: {GSPLAT_AVAILABLE}")
    
    model = GaussianSplattingSimple(64, 3, 256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    feat = torch.randn(2, 64, 32, 32).to(device)
    lr = torch.randn(2, 3, 32, 32).to(device)
    
    for scale in [2, 3, 4]:
        out = model(feat, lr, (32 * scale, 32 * scale))
        print(f"  {scale}x: {feat.shape} -> {out.shape}")
