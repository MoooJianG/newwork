[根目录](../CLAUDE.md) > **LDCSR**

---

# LDCSR - 潜在扩散连续尺度超分辨率

> **模块职责**: 基于潜在扩散模型的遥感图像连续尺度超分辨率
> **语言**: Python 3.10
> **框架**: PyTorch Lightning 1.7+
> **最后更新**: 2025-11-16 13:42:04

---

## 变更记录 (Changelog)

### 2025-11-16 13:42:04 - 初始化文档
- 完成模块架构分析
- 识别两阶段训练流程
- 建立目录结构映射

---

## 模块职责

LDCSR (Latent Diffusion for Continuous-Scale Super-Resolution) 实现了一个两阶段的遥感图像超分辨率系统：

1. **第一阶段**: 训练一个潜在自编码器（AutoencoderKL），将图像压缩到低维潜在空间
2. **第二阶段**: 在潜在空间中训练扩散模型（Diffusion UNet），实现连续尺度（1x-8x）的超分辨率

**核心创新**:
- 使用 E2SR-GAP（Geometry-Aware Projection）编码器/解码器
- 在潜在空间中进行扩散建模，降低计算成本
- 支持连续尺度因子（非离散倍数）

---

## 入口与启动

### 训练入口

**主入口**: `train.py`

```bash
# 第一阶段训练
python train.py --config configs/first_stage_kl_v6.yaml --gpus 0,1,2,3

# 第二阶段训练（需先完成第一阶段）
python train.py --config configs/second_stage_van_v4.yaml --gpus 0,1,2,3

# 从检查点恢复
python train.py --resume logs/path/to/checkpoint.ckpt
```

**关键参数**:
- `-c, --configs`: 配置文件路径（支持多个）
- `-r, --resume`: 从检查点恢复训练
- `-g, --gpus`: GPU 索引（逗号分隔）
- `-s, --seed`: 随机种子（默认23）
- `-l, --logdir`: 日志目录（默认 `logs/`）

### 测试入口

**主入口**: `test.py`

```bash
# 测试单个数据集
python test.py \
  --checkpoint logs/second_stage_van_v4/checkpoints/last.ckpt \
  --datasets AID \
  --scales 2,4,6,8 \
  --datatype HR_downsampled

# 测试第一阶段模型
python test.py --checkpoint <ckpt> --datasets AID --scales 4 --first_stage
```

**数据集类型**:
- `LRHR_paired`: LR/HR 配对数据
- `LR_only`: 仅 LR 输入（带伪 HR）
- `HR_only`: 仅 HR（单图像测试）
- `HR_downsampled`: 动态下采样 HR

---

## 对外接口

### 模型配置接口

**第一阶段配置** (`configs/first_stage_kl_v6.yaml`):

```yaml
model:
  target: models.first_stage_kl_atom.AutoencoderKL
  params:
    embed_dim: 4  # 潜在空间维度
    encoder_config:
      target: modules.e2sr.s1_v6.GAPEncoder
      params:
        ch: 64
        ch_mult: [1, 2, 2, 4]
    decoder_config:
      target: modules.e2sr.s1_v6.GAPDecoder
      params:
        num_sr_modules: [12, 12, 12, 12]
```

**第二阶段配置** (`configs/second_stage_van_v4.yaml`):

```yaml
model:
  target: models.second_stage_van_v4.S2Model
  params:
    timesteps: 4  # DDIM 步数
    unet_config:
      target: modules.diffusionmodules.openaimodel.UNetModel
      params:
        model_channels: 64
        attention_resolutions: [8]
    first_stage_config:
      ckpt_path: logs/first_stage_kl_v6/.../best.ckpt  # ← 关键依赖
```

### 数据集接口

**抽象基类**: `data/base.py`

```python
class BaseDataset(Dataset):
    def __getitem__(self, index):
        return {
            'hr': torch.Tensor,    # 高分辨率图像
            'lr': torch.Tensor,    # 低分辨率图像
            'file_name': str       # 文件名（用于保存结果）
        }
```

**实现类**:
- `DownsampledDataset` - 动态下采样数据集
- `MultiScaleDownsampledDataset` - 多尺度训练数据集
- `PairedImageDataset` - 预生成的 LR/HR 配对

### 模型推理接口

**第二阶段模型** (`models/second_stage_van_v4.py`):

```python
class S2Model(DDPM):
    def test_step(self, batch, batch_idx):
        """
        输入:
            batch = {
                'hr': [B, 3, H, W],
                'lr': [B, 3, h, w],
                'scale': [B],
                'file_name': List[str]
            }
        输出:
            {
                'sr': [B, 3, H, W],       # 超分辨率结果
                'sr_kd': [B, 3, H, W],    # 知识蒸馏版本
                'hr': [B, 3, H, W],       # 原始 HR
                'runtime': float          # 推理时间（秒）
            }
        """
```

---

## 关键依赖与配置

### Python 依赖

**核心库** (来自 `requirement.txt`):
```
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.7.0
omegaconf>=2.2.0          # 配置管理
pyiqa>=0.1.5              # 图像质量评估
lpips>=0.1.4              # 感知损失
pytorch-fid>=0.3.0        # FID 计算
einops>=0.5.0             # 张量重排
```

### 环境配置

```bash
# 推荐使用 Conda
conda create -n LDCSR python=3.10
conda activate LDCSR
pip install -r requirement.txt

# GPU 要求
# - 第一阶段: 4x GPU (batch_size=4, 每卡1张)
# - 第二阶段: 8x GPU (batch_size=8, 每卡1张)
# - 推理: 1x GPU (支持批量)
```

### 日志与检查点

**目录结构**:
```
logs/
├── first_stage_kl_v6/
│   └── 2024-10-11T22-27-56/
│       ├── checkpoints/
│       │   ├── last.ckpt
│       │   └── epoch=779-best.ckpt
│       ├── configs/
│       │   └── first_stage_kl_v6.yaml
│       └── tensorboard/
└── second_stage_van_v4/
    └── 2024-XX-XXT...
```

---

## 数据模型

### 数据流

```
原始 HR 图像 (AID/DOTA/DIOR)
    ↓ [data/prepare_split.py]
Train/Val/Test 分割
    ↓ [DataModule]
LR/HR 对 (动态下采样)
    ↓ [Normalize]
潜在编码
    ↓ [Diffusion]
潜在解码
    ↓ [Denormalize]
超分辨率图像
```

### 数据增强

**第一阶段** (`data/downsampled_dataset.py`):
- 随机裁剪：`lr_img_sz=48`
- 多尺度下采样：`min_scale=1, max_scale=8`
- 随机翻转/旋转（继承自 `transforms.py`）

**第二阶段** (`data/downsampled_dataset.py`):
- 固定 HR 尺寸：`hr_img_sz=256`
- 对应 LR 尺寸：根据 `scale` 动态计算

### 数据缓存策略

支持三种缓存模式（`cache` 参数）:
- `memory`: 全部加载到内存（推荐小数据集）
- `bin`: 预处理为二进制文件
- `none`: 实时读取（大数据集）

---

## 测试与质量

### 评估指标

**实现位置**: `metrics/`

| 指标 | 文件 | 说明 |
|-----|------|------|
| PSNR/SSIM | `psnr_ssim.py` | 传统图像质量指标 |
| FID | `fid.py` | 生成图像分布相似度 |
| LPIPS | `batched_iqa.py` | 感知相似度（VGG特征） |

**计算方式**:
```python
# PSNR/SSIM
from metrics.psnr_ssim import calc_psnr_ssim
psnr, ssim = calc_psnr_ssim(
    sr_np, hr_np,
    crop_border=math.ceil(scale),  # 边界裁剪
    test_Y=False                    # RGB or Y通道
)

# FID
from metrics import calc_fid
fid_score = calc_fid([sr_path, hr_path], device='cuda')

# LPIPS
import pyiqa
lpips_metric = pyiqa.create_metric('lpips').cuda()
lpips_score = lpips_metric(sr_tensor, hr_tensor)
```

### 基准测试

**脚本**: `auto_benchmark.py`

支持的数据集:
- AID（遥感航拍）
- DOTA（遥感目标检测）
- DIOR（遥感目标识别）

测试尺度: `2x, 3x, 4x, 5x, 6x, 7x, 8x`

---

## 常见问题 (FAQ)

### Q1: 第二阶段训练时找不到第一阶段检查点？

**A**: 修改 `configs/second_stage_van_v4.yaml` 中的 `ckpt_path`:
```yaml
first_stage_config:
  params:
    ckpt_path: logs/first_stage_kl_v6/YYYY-MM-DDTHH-MM-SS/checkpoints/epoch=XXX-best.ckpt
```

### Q2: OOM (Out of Memory) 错误？

**A**: 调整以下参数:
- 减小 `batch_size`（`configs/*.yaml`）
- 减小 `hr_img_sz` / `lr_img_sz`
- 减少 GPU 数量并相应调整 batch size

### Q3: 如何在自定义数据集上训练？

**A**:
1. 组织数据结构:
   ```
   dataset/YourData/
   ├── Train/HR/
   ├── Val/HR/
   └── Test/HR/
   ```
2. 修改配置文件的 `datapath` 参数
3. （可选）编写自定义 Dataset 类继承 `data/base.py`

### Q4: 支持哪些尺度因子？

**A**:
- **训练**: 连续尺度 1x-8x（通过 `min_scale` / `max_scale` 控制）
- **推理**: 任意尺度（包括分数尺度，如 2.5x）

### Q5: DDIM 采样步数如何选择？

**A**:
- `timesteps=4`: 快速推理（论文默认）
- `timesteps=10`: 平衡质量与速度
- `timesteps=50`: 最高质量（接近 DDPM）

---

## 相关文件清单

### 核心模型文件

```
models/
├── first_stage_kl.py           # 第一阶段基础 VAE
├── first_stage_kl_atom.py      # 第一阶段改进版（使用 E2SR）
├── second_stage_van_v4.py      # 第二阶段扩散模型（主版本）
├── second_stage_v*.py          # 第二阶段实验版本
├── diffusion/
│   ├── ddpm.py                 # DDPM 基类
│   └── ddim.py                 # DDIM 采样器
└── utils/
    └── optimizer.py            # 优化器工厂

modules/
├── e2sr/
│   └── s1_v6.py                # ★ GAPEncoder/GAPDecoder 核心
├── diffusionmodules/
│   ├── openaimodel.py          # ★ UNet 架构
│   └── util.py                 # 扩散工具函数
├── distributions/
│   └── distributions.py        # 高斯分布采样
└── image_degradation/
    └── bsrgan.py               # 数据增强降质

data/
├── base.py                     # ★ 数据集抽象基类
├── downsampled_dataset.py      # ★ 动态下采样数据集
├── paired_dataset.py           # 预生成配对数据集
├── datamodule.py               # Lightning DataModule
└── transforms.py               # 数据增强与归一化

losses/
├── contperceptual.py           # ★ LPIPS + GAN 损失（第一阶段）
├── basic_loss.py               # L1/L2/SSIM 损失
└── gan_loss.py                 # 对抗损失

configs/
├── first_stage_kl_v6.yaml      # ★ 第一阶段配置
└── second_stage_van_v4.yaml    # ★ 第二阶段配置
```

### 工具脚本

```
train.py                        # ★ 训练主入口
test.py                         # ★ 测试主入口
auto_benchmark.py               # 批量基准测试
gen_lr_datastes.py              # 生成 LR 数据集
data/prepare_split.py           # 数据集划分工具
```

---

## 变更记录 (Changelog)

### 2025-11-16 13:42:04 - 初始化文档
- 建立模块文档结构
- 记录两阶段训练流程
- 整理核心组件清单
- 补充常见问题解答
