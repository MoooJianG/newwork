#!/bin/bash
# ============================================================
# LDCSR 高斯查询超分训练脚本 (方案二)
# ============================================================
#
# 用法:
#   bash scripts/train_gaussian_query.sh [OPTIONS]
#
# 选项:
#   --gpus <gpu_ids>        GPU 索引（默认: 0,1,2,3）
#   --batch_size <size>     批大小（默认: 2）
#   --lr <learning_rate>    学习率（默认: 4.5e-6）
#   --resume <ckpt_path>    从检查点恢复
#   --debug                 调试模式（快速验证）
#
# 示例:
#   # 标准训练
#   bash scripts/train_gaussian_query.sh
#
#   # 单卡调试
#   bash scripts/train_gaussian_query.sh --gpus 0 --debug
#
#   # 从检查点恢复
#   bash scripts/train_gaussian_query.sh --resume logs/xxx/checkpoints/last.ckpt
#
# ============================================================

set -e  # 遇到错误立即退出

# ============================================================
# 参数解析
# ============================================================

GPUS="0,1,2,3"
BATCH_SIZE=2
LEARNING_RATE=4.5e-6
RESUME=""
DEBUG=false
SEED=23

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# 环境检查
# ============================================================

echo "=========================================="
echo " LDCSR 高斯查询训练启动"
echo "=========================================="
echo "GPU: $GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Random Seed: $SEED"
echo "Debug Mode: $DEBUG"
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "=========================================="
echo ""

# 检查配置文件
CONFIG_FILE="configs/first_stage_gaussian_query.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "✓ 配置文件: $CONFIG_FILE"

# 检查数据集
DATA_PATH="load/AID_split/train/HR"
if [ ! -d "$DATA_PATH" ]; then
    echo "⚠️  警告: 数据集路径不存在: $DATA_PATH"
    echo "   请确保已运行 data/prepare_split.py 准备数据集"
    read -p "   是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Python 环境
if ! python -c "import torch; import pytorch_lightning" 2>/dev/null; then
    echo "❌ 错误: Python 环境不完整"
    echo "   请运行: pip install -r requirement.txt"
    exit 1
fi
echo "✓ Python 环境检查通过"

# 检查高斯查询模块
if ! python -c "from modules.e2sr.gaussian_query import GaussianQueryModule" 2>/dev/null; then
    echo "❌ 错误: 高斯查询模块导入失败"
    echo "   请检查 modules/e2sr/gaussian_query.py 是否存在"
    exit 1
fi
echo "✓ 高斯查询模块检查通过"
echo ""

# ============================================================
# 训练配置调整
# ============================================================

EXTRA_ARGS=""

if [ "$DEBUG" = true ]; then
    echo "=========================================="
    echo " 调试模式 - 快速验证"
    echo "=========================================="
    EXTRA_ARGS="$EXTRA_ARGS --max_epochs 5"
    EXTRA_ARGS="$EXTRA_ARGS --limit_train_batches 10"
    EXTRA_ARGS="$EXTRA_ARGS --limit_val_batches 2"
    BATCH_SIZE=1
    echo "调整参数: max_epochs=5, batch_size=1, 仅10个训练批次"
    echo ""
fi

# 构建命令
CMD="python train.py \
    --config $CONFIG_FILE \
    --gpus $GPUS \
    --seed $SEED \
    --logdir logs/gaussian_query"

# 批大小和学习率
if [ "$BATCH_SIZE" != "2" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

if [ "$LEARNING_RATE" != "4.5e-6" ]; then
    CMD="$CMD --learning_rate $LEARNING_RATE"
fi

# 恢复训练
if [ -n "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "❌ 错误: 检查点文件不存在: $RESUME"
        exit 1
    fi
    CMD="$CMD --resume $RESUME"
fi

# 额外参数
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# ============================================================
# 训练前准备
# ============================================================

# 创建日志目录
mkdir -p logs/gaussian_query

# 打印完整命令
echo "=========================================="
echo " 训练命令"
echo "=========================================="
echo "$CMD"
echo ""

# 倒计时
echo "训练将在 3 秒后开始..."
for i in {3..1}; do
    echo "$i..."
    sleep 1
done
echo "开始训练！"
echo ""

# ============================================================
# 开始训练
# ============================================================

# 运行训练
eval $CMD

# 训练结束
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo ""
    echo "日志位置: logs/gaussian_query/"
    echo ""
    echo "下一步:"
    echo "  1. 查看训练曲线: tensorboard --logdir logs/gaussian_query"
    echo "  2. 测试模型: bash scripts/test_gaussian_query.sh --checkpoint logs/gaussian_query/.../best.ckpt"
    echo "  3. 可视化结果: python test.py --checkpoint <ckpt> --datasets AID --scales 2,4,6,8"
else
    echo "❌ 训练失败（退出码: $EXIT_CODE）"
    echo ""
    echo "可能的原因:"
    echo "  - GPU 内存不足: 尝试减小 batch_size 或 lr_img_sz"
    echo "  - 数据集路径错误: 检查 configs/first_stage_gaussian_query.yaml 中的 datapath"
    echo "  - 模块导入失败: 检查 modules/e2sr/gaussian_query.py"
    echo ""
    echo "调试建议:"
    echo "  bash scripts/train_gaussian_query.sh --debug --gpus 0"
fi
echo "=========================================="

exit $EXIT_CODE
