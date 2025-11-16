#!/bin/bash
# ============================================================
# LDCSR 高斯查询超分测试脚本 (方案二)
# ============================================================
#
# 用法:
#   bash scripts/test_gaussian_query.sh --checkpoint <ckpt_path> [OPTIONS]
#
# 必需参数:
#   --checkpoint <path>     检查点文件路径
#
# 可选参数:
#   --datasets <names>      数据集名称（默认: AID）
#                           支持: AID, DOTA, DIOR
#   --scales <scales>       测试尺度（默认: 2,4,6,8）
#                           支持任意尺度，包括分数尺度如: 2.5,3.7,5.2
#   --datatype <type>       数据类型（默认: HR_downsampled）
#                           选项: LRHR_paired, LR_only, HR_only, HR_downsampled
#   --save_dir <dir>        结果保存目录（默认: results/gaussian_query）
#   --gpu <id>              GPU 索引（默认: 0）
#   --batch_size <size>     批大小（默认: 1）
#   --first_stage           仅测试第一阶段（不使用扩散）
#   --timesteps <steps>     DDIM 步数（默认: 4）
#   --calc_fid              计算 FID 指标
#
# 示例:
#   # 基础测试（整数尺度）
#   bash scripts/test_gaussian_query.sh \
#     --checkpoint logs/gaussian_query/xxx/best.ckpt
#
#   # 任意尺度测试（包括分数尺度）
#   bash scripts/test_gaussian_query.sh \
#     --checkpoint logs/gaussian_query/xxx/best.ckpt \
#     --scales 2,2.5,3,3.7,4,5.2,6,8
#
#   # 多数据集测试
#   bash scripts/test_gaussian_query.sh \
#     --checkpoint logs/gaussian_query/xxx/best.ckpt \
#     --datasets AID,DOTA,DIOR \
#     --scales 2,4,6,8
#
#   # 第一阶段测试（跳过扩散）
#   bash scripts/test_gaussian_query.sh \
#     --checkpoint logs/gaussian_query/xxx/best.ckpt \
#     --first_stage \
#     --scales 2,3,4,5,6,7,8
#
# ============================================================

set -e

# ============================================================
# 参数解析
# ============================================================

CHECKPOINT=""
DATASETS="AID"
SCALES="2,4,6,8"
DATATYPE="HR_downsampled"
SAVE_DIR="results/gaussian_query"
GPU=0
BATCH_SIZE=1
FIRST_STAGE=false
TIMESTEPS=4
CALC_FID=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --scales)
            SCALES="$2"
            shift 2
            ;;
        --datatype)
            DATATYPE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --first_stage)
            FIRST_STAGE=true
            shift
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --calc_fid)
            CALC_FID=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# 参数验证
# ============================================================

echo "=========================================="
echo " LDCSR 高斯查询测试启动"
echo "=========================================="

if [ -z "$CHECKPOINT" ]; then
    echo "❌ 错误: 缺少 --checkpoint 参数"
    echo ""
    echo "用法: bash scripts/test_gaussian_query.sh --checkpoint <ckpt_path>"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 检查点文件不存在: $CHECKPOINT"
    exit 1
fi

echo "检查点: $CHECKPOINT"
echo "数据集: $DATASETS"
echo "尺度: $SCALES"
echo "数据类型: $DATATYPE"
echo "保存目录: $SAVE_DIR"
echo "GPU: $GPU"
echo "Batch Size: $BATCH_SIZE"
echo "第一阶段模式: $FIRST_STAGE"
if [ "$FIRST_STAGE" = false ]; then
    echo "DDIM Steps: $TIMESTEPS"
fi
echo "计算 FID: $CALC_FID"
echo "=========================================="
echo ""

# 检查环境
if ! python -c "import torch; import pytorch_lightning" 2>/dev/null; then
    echo "❌ 错误: Python 环境不完整"
    exit 1
fi
echo "✓ Python 环境检查通过"

if ! python -c "from modules.e2sr.gaussian_query import GaussianQueryModule" 2>/dev/null; then
    echo "❌ 错误: 高斯查询模块导入失败"
    exit 1
fi
echo "✓ 高斯查询模块检查通过"
echo ""

# ============================================================
# 运行测试
# ============================================================

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=$GPU python test.py \
    --checkpoint $CHECKPOINT \
    --datasets $DATASETS \
    --scales $SCALES \
    --datatype $DATATYPE \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE"

if [ "$FIRST_STAGE" = true ]; then
    CMD="$CMD --first_stage"
else
    CMD="$CMD --timesteps $TIMESTEPS"
fi

if [ "$CALC_FID" = true ]; then
    CMD="$CMD --calc_fid"
fi

echo "=========================================="
echo " 测试命令"
echo "=========================================="
echo "$CMD"
echo ""

echo "测试开始..."
echo ""

# 运行测试
eval $CMD

EXIT_CODE=$?

# ============================================================
# 测试结果汇总
# ============================================================

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成！"
    echo ""
    echo "结果保存位置: $SAVE_DIR"
    echo ""

    # 查找指标文件
    METRICS_FILE=$(find "$SAVE_DIR" -name "metrics_*.txt" 2>/dev/null | head -1)

    if [ -f "$METRICS_FILE" ]; then
        echo "=========================================="
        echo " 测试指标汇总"
        echo "=========================================="
        cat "$METRICS_FILE"
        echo "=========================================="
    fi

    echo ""
    echo "下一步:"
    echo "  1. 查看详细指标: cat $SAVE_DIR/metrics_*.txt"
    echo "  2. 可视化结果: ls $SAVE_DIR/*.png"
    echo "  3. 对比baseline: bash scripts/compare_results.sh"

    # 分数尺度测试建议
    if [[ "$SCALES" == *"."* ]]; then
        echo ""
        echo "💡 提示: 检测到分数尺度测试"
        echo "   这是高斯查询方案的独特优势！"
        echo "   传统方法无法在分数尺度上工作。"
    fi

else
    echo "❌ 测试失败（退出码: $EXIT_CODE）"
    echo ""
    echo "可能的原因:"
    echo "  - 检查点文件损坏"
    echo "  - 数据集路径错误"
    echo "  - GPU 内存不足"
    echo "  - 模型架构不匹配"
    echo ""
    echo "调试建议:"
    echo "  1. 检查检查点是否为高斯查询版本:"
    echo "     grep 'use_gaussian_query' logs/gaussian_query/.../configs/*.yaml"
    echo "  2. 测试单张图像:"
    echo "     python test.py --checkpoint <ckpt> --datasets AID --datatype HR_only"
    echo "  3. 使用第一阶段模式:"
    echo "     bash scripts/test_gaussian_query.sh --checkpoint <ckpt> --first_stage"
fi
echo "=========================================="

exit $EXIT_CODE
