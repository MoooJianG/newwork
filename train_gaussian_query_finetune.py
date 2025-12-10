import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import argparse
import sys

sys.path.insert(0, '/root/autodl-tmp/newwork')

from models.first_stage_kl_atom import AutoencoderKL


def load_and_freeze_encoder_decoder_base(model, pretrained_path):
    """加载预训练 encoder + decoder基础部分，并冻结"""
    print(f"\n{'='*80}")
    print(f"加载预训练权重: {pretrained_path}")
    print(f"{'='*80}")

    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # 提取 encoder + decoder基础部分（排除upsampler）
    pretrained_state = {}
    for key, value in state_dict.items():
        # 加载 encoder 全部
        if key.startswith('encoder.'):
            pretrained_state[key] = value
        # 加载 decoder，但排除 upsampler
        elif key.startswith('decoder.') and not key.startswith('decoder.upsampler'):
            pretrained_state[key] = value

    # 加载参数（upsampler会missing，这是预期的）
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_state, strict=False)

    print(f"\n✅ 加载了 {len(pretrained_state)} 个预训练参数")
    print(f"   Encoder: {len([k for k in pretrained_state if k.startswith('encoder.')])}")
    print(f"   Decoder (不含upsampler): {len([k for k in pretrained_state if k.startswith('decoder.')])}")

    # 检查missing的是否都是upsampler或loss（预期的）
    missing_upsampler = [k for k in missing_keys if 'upsampler' in k]
    missing_others = [k for k in missing_keys if 'upsampler' not in k and 'loss' not in k and 'quant' not in k]

    print(f"\n   Missing upsampler keys (预期): {len(missing_upsampler)}")
    if missing_others:
        print(f"   ⚠️  Missing other keys (不应该有): {len(missing_others)}")
        for k in missing_others[:5]:
            print(f"      {k}")

    # 冻结 encoder + decoder基础部分
    frozen_params = 0
    trainable_params_count = 0

    for name, param in model.named_parameters():
        # 冻结encoder和decoder基础部分（不含upsampler）
        if name.startswith('encoder.') or \
           (name.startswith('decoder.') and not name.startswith('decoder.upsampler')):
            param.requires_grad = False
            frozen_params += 1
        else:
            trainable_params_count += 1

    print(f"\n🔒 冻结了 {frozen_params} 个参数（Encoder + Decoder基础部分）")
    print(f"🎯 可训练 {trainable_params_count} 个参数（仅 GaussianQuery Upsampler）")

    # 统计可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n📊 参数统计:")
    print(f"   可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   总参数: {total_params:,}")
    print(f"{'='*80}\n")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--pretrained', type=str, required=True, help='预训练 checkpoint 路径')
    parser.add_argument('--gpus', type=str, default='0', help='GPU 索引')
    parser.add_argument('--seed', type=int, default=23, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    pl.seed_everything(args.seed)

    # 加载配置
    config = OmegaConf.load(args.config)

    # 解析 GPU
    gpu_list = [int(x) for x in args.gpus.split(',')]
    print(f"使用 GPU: {gpu_list}")

    # 创建模型
    model = AutoencoderKL(**config.model.params)

    # 加载预训练 encoder + decoder基础部分，并冻结
    model = load_and_freeze_encoder_decoder_base(model, args.pretrained)

    # 创建数据模块
    from data.datamodule import DataModuleFromConfig
    data = DataModuleFromConfig(**config.data.params)

    # 创建 Trainer
    trainer_config = OmegaConf.to_container(config.lightning.trainer, resolve=True)
    trainer_config['accelerator'] = 'gpu'
    trainer_config['devices'] = gpu_list

    trainer = pl.Trainer(**trainer_config)

    # 开始训练
    print(f"\n{'='*80}")
    print("开始训练（仅训练 GaussianQuery Upsampler）")
    print(f"{'='*80}\n")

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
