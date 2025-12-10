# 解码器引入SR分支和emb的解码分支。
# s1_v7: 添加 GaussianSplattingV2 支持
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.upsampler import SADNUpsampler
from .common import (
    ResnetBlock,
    Downsample,
    Normalize,
    nonlinearity,
    FRU,
    StyleLayer_norm_scale_shift,
)

try:
    from .gaussian_query import GaussianQueryDecoder
    GAUSSIAN_QUERY_AVAILABLE = True
except ImportError:
    GAUSSIAN_QUERY_AVAILABLE = False
    print("高斯查询模块未找到，use_gaussian_query 将被忽略")

try:
    from .gaussian_splatting_v2 import GaussianSplattingDecoderV2
    GAUSSIAN_V2_AVAILABLE = True
except ImportError:
    GAUSSIAN_V2_AVAILABLE = False
    print("GaussianSplattingV2 模块未找到，use_gaussian_v2 将被忽略")


class GAPEncoder(nn.Module):
    # 与LDM一致
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=6,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hr, lr):
        temb = None

        if hr is not None:
            lr_up = F.interpolate(
                lr, size=hr.shape[2:], mode="bicubic", align_corners=False
            )
            _input = torch.cat([lr_up, hr], dim=1)
        else:
            _input = lr

        # downsampling
        hs = [self.conv_in(_input)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class GAPDecoder(nn.Module):
    # 想办法把gap embedding 放进来
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        in_channels,
        z_channels,
        num_sr_modules=[8, 8, 8, 8],
        use_gaussian_query=False,
        num_gaussians=100,
        gaussian_kernel_size=5,
        gaussian_hidden_dim=256,
        gaussian_unfold_row=7,
        gaussian_unfold_column=7,
        # GaussianSplattingV2 参数
        use_gaussian_v2=False,
        gaussian_v2_hidden_dim=256,
        gaussian_v2_dict_size=729,
        gaussian_v2_use_gsplat=False,
        gaussian_v2_upsample_factor=2,
        **ignorekwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.use_gaussian_query = use_gaussian_query and GAUSSIAN_QUERY_AVAILABLE
        self.use_gaussian_v2 = use_gaussian_v2 and GAUSSIAN_V2_AVAILABLE

        # 互斥检查：GaussianQuery 和 GaussianV2 不能同时启用
        if self.use_gaussian_query and self.use_gaussian_v2:
            print("⚠️  警告: use_gaussian_query 和 use_gaussian_v2 不能同时启用")
            print("    优先使用 GaussianV2，禁用 GaussianQuery")
            self.use_gaussian_query = False

        assert len(num_sr_modules) == len(ch_mult)
        self.style = StyleLayer_norm_scale_shift()

        ############# sr branch #############
        self.num_sr_modules = num_sr_modules
        self.head = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.FRUList = nn.ModuleList()
        for num in num_sr_modules:
            self.FRUList.append(FRU(ch, num_modules=num))

        # 上采样器选择：GaussianV2 > GaussianQuery > SADN
        if self.use_gaussian_v2:
            self.upsampler = GaussianSplattingDecoderV2(
                in_channels=ch,
                out_channels=out_ch,
                hidden_dim=gaussian_v2_hidden_dim,
                dict_size=gaussian_v2_dict_size,
                use_gsplat=gaussian_v2_use_gsplat,
                upsample_factor=gaussian_v2_upsample_factor
            )
            print(f"✅ 使用 GaussianSplattingV2 解码器")
            print(f"   - hidden_dim: {gaussian_v2_hidden_dim}")
            print(f"   - dict_size: {gaussian_v2_dict_size}")
            print(f"   - use_gsplat: {gaussian_v2_use_gsplat}")
            print(f"   - upsample_factor: {gaussian_v2_upsample_factor}")
        elif self.use_gaussian_query:
            self.upsampler = GaussianQueryDecoder(
                in_channels=ch,
                out_channels=out_ch,
                num_gaussians=num_gaussians,
                kernel_size=gaussian_kernel_size,
                hidden_dim=gaussian_hidden_dim,
                unfold_row=gaussian_unfold_row,
                unfold_column=gaussian_unfold_column
            )
            print(f"✅ 使用 GaussianQuery 解码器 (num_gaussians={num_gaussians})")
        else:
            self.upsampler = SADNUpsampler(ch, kSize=3, out_channels=out_ch)
            print("✅ 使用传统 SADN 上采样器")

        ############# emb branch #############
        # z to block_in
        block_in = ch * ch_mult[-1]
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # up
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.transfer = nn.Conv2d(block_in, self.ch, 1)
            up.condition_scale1 = nn.Linear(1, self.ch)
            up.condition_scale2 = nn.Linear(1, self.ch)
            self.up.insert(0, up)  # prepend to get consistent order

    def get_last_layer(self):
        if self.use_gaussian_v2:
            # GaussianSplattingV2 的最后一层是 color_mlp
            return self.upsampler.gaussian_splatting.color_mlp[-1].weight
        elif self.use_gaussian_query:
            # 高斯查询解码器的最后一层是 MLP
            return self.upsampler.gaussian_query.mlp[-1].weight
        else:
            return self.upsampler.fuse[-1].weight

    def forward(self, emb, lr, out_size):
        lr_size = lr.shape[2:]
        scale = torch.tensor([out_size[0] * 1.0 / lr_size[0]]).to(lr.device)

        ############# emb branch #############
        temb = None
        # z to block_in
        h = self.conv_in(emb)
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # upsampling but without upsample
        h_list, scales1, scales2 = [], [], []
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
            h_transfer = self.up[i_level].transfer(h)
            h_list.append(F.interpolate(h_transfer, lr_size, mode="nearest"))
            scale1 = self.up[i_level].condition_scale1(scale)
            scale2 = self.up[i_level].condition_scale2(scale)
            scales1.append(scale1.clone())
            scales2.append(scale2.clone())

        ############# sr branch #############
        head = self.head(lr)
        mid = head
        for i_level in range(self.num_resolutions):
            _input = self.style(
                mid, scales1[i_level], scales2[i_level], h_list[i_level]
            )
            mid = self.FRUList[i_level](_input)

        if self.use_gaussian_v2:
            # GaussianSplattingV2 需要 (feat, lr, out_size)
            out = self.upsampler(mid, lr, out_size)
        elif self.use_gaussian_query:
            # GaussianQueryDecoder 需要 (feat, lr, out_size)
            out = self.upsampler(mid, lr, out_size)
        else:
            # SADNUpsampler 需要 (feat, out_size)
            x_up = F.interpolate(lr, out_size, mode="bicubic", align_corners=False)
            out = self.upsampler(mid, out_size) + x_up

        return out
