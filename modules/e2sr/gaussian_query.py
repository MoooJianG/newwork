import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_utils import (
    make_coord, make_cell, generate_meshgrid,
    fetching_features_from_tensor, batched_grid_sample
)


class GaussianQueryModule(nn.Module):
    """
    高斯散射查询模块 - 替换传统 Upsampler

    通过 2D 高斯散射 + 坐标查询实现任意尺度超分辨率

    Args:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出通道数（通常为 3）
        num_gaussians (int): 高斯核候选数量
        kernel_size (int): 高斯核栅格大小
        hidden_dim (int): MLP 隐藏层维度
        unfold_row (int): Unfold 行大小（用于分块处理）
        unfold_column (int): Unfold 列大小
    """

    def __init__(
        self,
        in_channels,
        out_channels=3,
        num_gaussians=100,
        kernel_size=5,
        hidden_dim=256,
        unfold_row=7,
        unfold_column=7
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_gaussians = num_gaussians
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.unfold_row = unfold_row
        self.unfold_column = unfold_column

        # 高斯分类器 - 为每个像素分配高斯核权重
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_gaussians, 1)
        )

        # 根据 num_gaussians 计算网格大小
        grid_size = int(num_gaussians ** 0.5)
        if grid_size * grid_size != num_gaussians:
            grid_size = int((num_gaussians ** 0.5) + 0.5)
            actual_num_gaussians = grid_size * grid_size
            if actual_num_gaussians != num_gaussians:
                print(f"num_gaussians={num_gaussians} 不是完全平方数,调整为 {actual_num_gaussians} ({grid_size}x{grid_size})")
                self.num_gaussians = actual_num_gaussians

        sigma_x, sigma_y = torch.meshgrid(
            torch.linspace(0.2, 3.0, grid_size),
            torch.linspace(0.2, 3.0, grid_size),
            indexing='ij'
        )
        self.sigma_x = nn.Parameter(sigma_x.reshape(-1))  # [num_gaussians]
        self.sigma_y = nn.Parameter(sigma_y.reshape(-1))
        self.opacity = nn.Parameter(torch.sigmoid(torch.ones(self.num_gaussians, 1)))
        self.rho = nn.Parameter(torch.clamp(torch.zeros(self.num_gaussians, 1), -1, 1))

        # 特征增强模块
        self.coef = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)

        # MLP 解码器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels)
        )

        self.unfold = nn.Unfold(
            kernel_size=(self.unfold_row, self.unfold_column),
            stride=(self.unfold_row, self.unfold_column)
        )
        # Fold 对象需要在 forward 时根据动态尺寸创建，缓存参数
        self._fold_cache = {}  # 缓存不同尺寸的 Fold 对象

    def weighted_gaussian_parameters(self, logits):
        """
        计算加权高斯参数

        Args:
            logits: [B, num_gaussians, H, W] - 分类器输出

        Returns:
            (weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho)
            每个都是 [H*W] 的张量（对batch维度取平均）
        """
        B, C, H, W = logits.size()

        # Softmax 归一化为概率分布
        logits = F.softmax(logits, dim=1)
        logits = logits.permute(0, 2, 3, 1)  # [B, H, W, num_gaussians]

        # 加权求和
        weighted_sigma_x = (logits * self.sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_sigma_y = (logits * self.sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_opacity = (logits * self.opacity[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_rho = (logits * self.rho[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)


        weighted_sigma_x = weighted_sigma_x.reshape(B, -1).mean(dim=0)  # [H*W]
        weighted_sigma_y = weighted_sigma_y.reshape(B, -1).mean(dim=0)
        weighted_opacity = weighted_opacity.reshape(B, -1).mean(dim=0)
        weighted_rho = weighted_rho.reshape(B, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho

    def generate_gaussian_splatting(self, feat, logits, target_size, scale):
        """
        生成高斯散射特征场

        Args:
            feat: [B, C, H, W] - 输入特征
            logits: [B, num_gaussians, H, W] - 高斯分类 logits
            target_size: (H_out, W_out) - 目标尺寸
            scale: float - 尺度因子

        Returns:
            gaussian_feat: [B, C, H_out, W_out] - 高斯散射后的特征场
        """
        B, C, H, W = feat.shape
        device = feat.device

        # 1. 上采样到目标尺寸的块对齐版本
        num_kernels_row = math.ceil(H / self.unfold_row)
        num_kernels_column = math.ceil(W / self.unfold_column)
        upsampled_size = (num_kernels_row * self.unfold_row, num_kernels_column * self.unfold_column)

        upsampled_feat = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)

        # 2. Unfold 分块处理
        unfolded_feat = self.unfold(upsampled_feat)  # [B, C*row*col, L]
        unfolded_logits = self.unfold(upsampled_logits)  # [B, num_g*row*col, L]

        L = unfolded_feat.shape[-1]  

        # 重塑为 [B*L, C, row, col]
        unfolded_feat = unfolded_feat.transpose(1, 2).reshape(B * L, C, self.unfold_row, self.unfold_column)
        unfolded_logits = unfolded_logits.transpose(1, 2).reshape(
            B * L, self.num_gaussians, self.unfold_row, self.unfold_column
        )

        # 3. 生成颜色和坐标
        coords = generate_meshgrid(self.unfold_row, self.unfold_column, device)
        num_LR_points = self.unfold_row * self.unfold_column
        colors, coords_norm = fetching_features_from_tensor(unfolded_feat, coords)  # [B*L, N, C], [N, 2]

        # 4. 计算加权高斯参数
        weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho = \
            self.weighted_gaussian_parameters(unfolded_logits)

        # weighted_* 的形状是 [num_LR_points]
        # 所有batch共享相同的高斯参数
        batch_size = B * L
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1)  # [num_LR_points, 1, 1]
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rho = weighted_rho.view(num_LR_points, 1, 1)

        # 5. 生成高斯核
        total_points = num_LR_points  

        # 5.1 协方差矩阵 - 
        covariance = torch.stack([
            torch.stack([sigma_x ** 2 + 1e-5, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2 + 1e-5], dim=-1)
        ], dim=-2)  # [total_points, 2, 2]

        inv_covariance = torch.inverse(covariance)

        # 5.2 生成栅格 [-5, 5]
        ax = torch.linspace(-5.0, 5.0, steps=self.kernel_size, device=device)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')
        xy = torch.stack([xx, yy], dim=-1)  # [k, k, 2]

        # 5.3 计算高斯值（马氏距离）- 使用einsum
        # xy: [k, k, 2], inv_covariance: [num_LR_points, 2, 2]
        # 使用Einstein求和约定计算: z = -0.5 * xy @ inv_covariance @ xy^T
        z = torch.einsum('...i,...ij,...j->...', xy, -0.5 * inv_covariance, xy)  # [num_LR_points, k, k]

        kernel = torch.exp(z) / (
            2 * np.pi *
            torch.sqrt(torch.det(covariance)).view(total_points, 1, 1)
        )

        # 归一化
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)
        kernel_normalized = kernel / kernel_max_2  # [num_LR_points, k, k]

        # 扩展到通道维度 [num_LR_points, C, k, k]
        kernel_reshaped = kernel_normalized.repeat(1, C, 1).contiguous().view(
            num_LR_points * C, self.kernel_size, self.kernel_size
        )
        kernel_color = kernel_reshaped.unsqueeze(0).reshape(
            num_LR_points, C, self.kernel_size, self.kernel_size
        )

        # 6. 添加填充
        kernel_h = round(self.unfold_row * scale)
        kernel_w = round(self.unfold_column * scale)

        pad_h = kernel_h - self.kernel_size
        pad_w = kernel_w - self.kernel_size

        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"Kernel size {self.kernel_size} 过大，无法填充到 ({kernel_h}, {kernel_w})")

        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = F.pad(kernel_color, padding, "constant", 0) 

        # 7. Affine Transform 平移到像素中心
        batch_size = B * L
        b, c, h, w = kernel_color_padded.shape
        theta = torch.zeros(batch_size, b, 2, 3, dtype=torch.float32, device=device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm.unsqueeze(0).expand(batch_size, -1, -1)

        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * b, c, h, w], align_corners=True)
        kernel_color_padded_expanded = kernel_color_padded.repeat(batch_size, 1, 1, 1).contiguous()
        kernel_color_padded_translated = F.grid_sample(
            kernel_color_padded_expanded,
            grid.contiguous(),
            align_corners=True
        )
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, b, c, h, w)

        # 8. Splatting 累加
        # weighted_opacity 形状是 [num_LR_points]，需要扩展到 [batch_size, num_LR_points]
        colors_weighted = colors * weighted_opacity.unsqueeze(-1).expand(batch_size, -1, -1)  # [batch_size, num_LR_points, C]
        color_values_reshaped = colors_weighted.unsqueeze(-1).unsqueeze(-1)  # [B*L, N, C, 1, 1]
        final_image_layers = color_values_reshaped * kernel_color_padded_translated  # [B*L, N, C, h, w]
        final_image = final_image_layers.sum(dim=1)  # [B*L, C, h, w]
        final_image = torch.clamp(final_image, 0, 1)

        output_size = (kernel_h * num_kernels_row, kernel_w * num_kernels_column)
        fold_key = (output_size, kernel_h, kernel_w)

        if fold_key not in self._fold_cache:
            self._fold_cache[fold_key] = nn.Fold(
                output_size=output_size,
                kernel_size=(kernel_h, kernel_w),
                stride=(kernel_h, kernel_w)
            ).to(device)
        fold = self._fold_cache[fold_key]

        final_image = final_image.reshape(B, L, C * kernel_h * kernel_w).transpose(1, 2)
        final_image = fold(final_image)  # [B, C, H_full, W_full]

        # 10. 插值到精确目标尺寸
        final_image = F.interpolate(final_image, size=target_size, mode='bicubic', align_corners=False)

        return final_image

    def query_rgb(self, feat, gaussian_feat, coord, scale, cell):
        """
        通过坐标查询获取 RGB 值（带局部位置编码）

        Args:
            feat: [B, C, H, W] - 原始 LR 特征
            gaussian_feat: [B, C, H_out, W_out] - 高斯散射后的特征场
            coord: [B, Q, 2] - 查询坐标（归一化 [-1, 1]）
            scale: [B] 或 float - 尺度因子
            cell: [B, Q, 2] - 像素单元大小

        Returns:
            rgb: [B, Q, 3] - 查询的 RGB 值
        """
        B, C, H, W = feat.shape
        _, _, H_out, W_out = gaussian_feat.shape

        # 1. 计算频率和系数
        coef = self.coef(gaussian_feat)  # [B, hidden_dim, H_out, W_out]
        freq = self.freq(gaussian_feat)

        # 2. 生成特征坐标
        feat_coord = make_coord((H_out, W_out), flatten=False).to(feat.device)  # [H_out, W_out, 2]
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(B, 2, H_out, W_out)

        # 3. Grid Sample 查询
        coord_ = coord.clone()
        q_coef = F.grid_sample(
            coef,
            coord_.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, Q, hidden_dim]

        q_freq = F.grid_sample(
            freq,
            coord_.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        q_coord = F.grid_sample(
            feat_coord,
            coord_.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, Q, 2]

        # 4. 计算相对坐标
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= H_out
        rel_coord[:, :, 1] *= W_out

        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= H_out
        rel_cell[:, :, 1] *= W_out

        # 5. 局部位置编码
        bs, q = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)  # [B, Q, hidden_dim/2, 2]
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))  # [B, Q, hidden_dim/2, 2]
        q_freq = torch.sum(q_freq, dim=-2)  # [B, Q, 2]
        q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
        q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

        # 6. 组合特征
        inp = torch.mul(q_coef, q_freq)  # [B, Q, hidden_dim]

        # 7. MLP 解码
        rgb = self.mlp(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)  # [B, Q, 3]

        return rgb

    def forward(self, feat, coord, scale, cell):
        """
        前向传播 - 完整的高斯散射 + 坐标查询流程

        Args:
            feat: [B, C, H, W] - 输入特征
            coord: [B, Q, 2] - 查询坐标（归一化 [-1, 1]）
            scale: [B] 或 float - 尺度因子
            cell: [B, Q, 2] - 像素单元大小

        Returns:
            rgb: [B, Q, 3] - 查询的 RGB 值
        """
        B, C, H, W = feat.shape

        # 提取尺度值
        if isinstance(scale, torch.Tensor):
            scale_val = float(scale[0])
        else:
            scale_val = float(scale)

        # 计算目标尺寸
        target_size = (round(H * scale_val), round(W * scale_val))

        # 1. 高斯分类
        logits = self.classifier(feat)  # [B, num_gaussians, H, W]

        # 2. 高斯散射生成特征场
        gaussian_feat = self.generate_gaussian_splatting(
            feat, logits, target_size, scale_val
        )  # [B, C, H_out, W_out]

        # 3. 坐标查询
        rgb = self.query_rgb(feat, gaussian_feat, coord, scale, cell)  # [B, Q, 3]

        return rgb


class GaussianQueryDecoder(nn.Module):
    """
    专用于 LDCSR 的高斯查询解码器
    替换原有的 SADNUpsampler

    Args:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出通道数
        **gaussian_kwargs: 传递给 GaussianQueryModule 的参数
    """

    def __init__(self, in_channels, out_channels=3, **gaussian_kwargs):
        super().__init__()
        self.gaussian_query = GaussianQueryModule(
            in_channels=in_channels,
            out_channels=out_channels,
            **gaussian_kwargs
        )

    def forward(self, feat, lr, out_size):
        """
        解码器前向传播

        Args:
            feat: [B, C, H, W] - SR 分支的特征
            lr: [B, 3, h, w] - 低分辨率图像（用于残差）
            out_size: (H_out, W_out) - 目标尺寸

        Returns:
            out: [B, 3, H_out, W_out] - 超分结果
        """
        B = feat.shape[0]
        lr_size = lr.shape[2:]
        scale = torch.tensor([out_size[0] * 1.0 / lr_size[0]]).to(feat.device)

        # 1. 生成查询坐标
        coord = make_coord(out_size, flatten=True).to(feat.device)  # [H*W, 2]
        cell = make_cell(coord, out_size)  # [H*W, 2]

        # 2. 扩展到批次维度
        coord = coord.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
        cell = cell.unsqueeze(0).expand(B, -1, -1)

        # 3. 高斯查询
        rgb = self.gaussian_query(feat, coord, scale, cell)  # [B, H*W, 3]

        # 4. 重塑为图像格式
        out = rgb.view(B, out_size[0], out_size[1], 3).permute(0, 3, 1, 2)

        # 5. 残差连接
        x_up = F.interpolate(lr, out_size, mode='bicubic', align_corners=False)
        out = out + x_up

        return out


# ============ 测试代码 ============
if __name__ == '__main__':
    print("========== 测试高斯散射查询模块 ==========\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 测试 1: GaussianQueryModule
    print("测试 1: GaussianQueryModule")
    model = GaussianQueryModule(
        in_channels=64,
        out_channels=3,
        num_gaussians=100,
        kernel_size=5,
        hidden_dim=256,
        unfold_row=7,
        unfold_column=7
    ).to(device)

    feat = torch.rand(2, 64, 16, 16).to(device)
    scale = torch.tensor([2.0]).to(device)
    target_size = (32, 32)

    coord = make_coord(target_size, flatten=True).to(device)
    cell = make_cell(coord, target_size).to(device)
    coord = coord.unsqueeze(0).expand(2, -1, -1)
    cell = cell.unsqueeze(0).expand(2, -1, -1)

    print(f"  输入: feat {feat.shape}, scale {scale.item()}")
    print(f"  查询: coord {coord.shape}, cell {cell.shape}")

    rgb = model(feat, coord, scale, cell)
    print(f"  输出: rgb {rgb.shape}")
    print(f"  RGB 范围: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]\n")

    # 测试 2: GaussianQueryDecoder
    print("测试 2: GaussianQueryDecoder")
    decoder = GaussianQueryDecoder(
        in_channels=64,
        out_channels=3,
        num_gaussians=100,
        kernel_size=5
    ).to(device)

    feat = torch.rand(2, 64, 16, 16).to(device)
    lr = torch.rand(2, 3, 16, 16).to(device)
    out_size = (64, 64)

    print(f"  输入: feat {feat.shape}, lr {lr.shape}")
    print(f"  目标尺寸: {out_size}")

    out = decoder(feat, lr, out_size)
    print(f"  输出: out {out.shape}")
    print(f"  输出范围: [{out.min().item():.3f}, {out.max().item():.3f}]\n")

    # 测试 3: 任意尺度
    print("测试 3: 任意尺度测试")
    scales = [2.0, 2.5, 3.0, 3.7, 4.0]
    for s in scales:
        target_h = round(16 * s)
        target_w = round(16 * s)
        out = decoder(feat, lr, (target_h, target_w))
        print(f"  尺度 {s}x: 输出 {out.shape}")

