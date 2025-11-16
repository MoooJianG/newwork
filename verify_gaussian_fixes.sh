#!/bin/bash
# ============================================================
# LDCSR 高斯查询方案 - 修复验证脚本
# ============================================================
#
# 验证所有高斯查询相关的修复是否已正确应用
#
# 使用方法: bash verify_gaussian_fixes.sh
#
# ============================================================

echo "=========================================="
echo " LDCSR 高斯查询修复验证"
echo "=========================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试函数
check_fix() {
    local name="$1"
    local file="$2"
    local pattern="$3"
    local description="$4"

    echo -n "检查 $name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ 文件不存在${NC}"
        echo "  文件路径: $file"
        ((FAIL_COUNT++))
        return 1
    fi

    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓ 通过${NC}"
        echo "  $description"
        ((PASS_COUNT++))
        return 0
    else
        echo -e "${RED}✗ 失败${NC}"
        echo "  未找到: $pattern"
        echo "  文件: $file"
        ((FAIL_COUNT++))
        return 1
    fi
}

echo "=== 核心文件检查 ==="
echo ""

# 检查 1: gaussian_utils.py 存在性
check_fix "gaussian_utils.py" \
    "modules/e2sr/gaussian_utils.py" \
    "def make_coord" \
    "坐标生成工具已就位"

# 检查 2: gaussian_query.py 存在性
check_fix "gaussian_query.py" \
    "modules/e2sr/gaussian_query.py" \
    "class GaussianQueryDecoder" \
    "高斯查询解码器已定义"

# 检查 3: s1_v6.py 导入修复
check_fix "s1_v6.py 导入" \
    "modules/e2sr/s1_v6.py" \
    "from .gaussian_query import GaussianQueryDecoder" \
    "高斯查询模块已导入"

echo ""
echo "=== 关键修复检查 ==="
echo ""

# 检查 4: s1_v6.py 接口修复
check_fix "GAPDecoder 接口修复" \
    "modules/e2sr/s1_v6.py" \
    "if self.use_gaussian_query:" \
    "条件分支已添加（修复参数不匹配）"

# 检查 5: einsum 维度修复
check_fix "einsum 维度修复" \
    "modules/e2sr/gaussian_query.py" \
    "xy_flat = xy.reshape(num_LR_points, -1, 2)" \
    "einsum 已替换为显式矩阵乘法"

echo ""
echo "=== 配置文件检查 ==="
echo ""

# 检查 6: 配置文件存在
check_fix "训练配置" \
    "configs/first_stage_gaussian_query.yaml" \
    "use_gaussian_query: true" \
    "高斯查询配置已启用"

echo ""
echo "=== 脚本文件检查 ==="
echo ""

# 检查 7: 训练脚本
check_fix "训练脚本" \
    "scripts/train_gaussian_query.sh" \
    "first_stage_gaussian_query.yaml" \
    "训练脚本已配置"

# 检查 8: 测试脚本
check_fix "测试脚本" \
    "scripts/test_gaussian_query.sh" \
    "use_gaussian_query" \
    "测试脚本已配置"

echo ""
echo "=== Python 语法检查 ==="
echo ""

# 语法检查
for file in \
    "modules/e2sr/gaussian_utils.py" \
    "modules/e2sr/gaussian_query.py" \
    "modules/e2sr/s1_v6.py"; do

    if [ -f "$file" ]; then
        echo -n "检查 $file 语法... "
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "${GREEN}✓ 通过${NC}"
            ((PASS_COUNT++))
        else
            echo -e "${RED}✗ 语法错误${NC}"
            ((FAIL_COUNT++))
        fi
    fi
done

echo ""
echo "=========================================="
echo " 验证结果"
echo "=========================================="
echo ""
echo -e "${GREEN}通过: $PASS_COUNT${NC}"
echo -e "${RED}失败: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有检查通过！代码已准备好部署到云服务器。${NC}"
    echo ""
    echo "下一步："
    echo "  1. 同步代码到云服务器:"
    echo "     git push origin main  # (如果使用 Git)"
    echo "     或"
    echo "     scp -r modules/ root@server:/path/to/LDCSR/"
    echo ""
    echo "  2. 在云服务器上运行训练:"
    echo "     bash scripts/train_gaussian_query.sh --gpus 0"
    exit 0
else
    echo -e "${RED}❌ 有 $FAIL_COUNT 项检查失败，请先修复这些问题。${NC}"
    echo ""
    echo "建议操作："
    echo "  1. 检查失败的文件是否存在"
    echo "  2. 运行对应的快速修复脚本:"
    echo "     bash quick_fix.sh           # 修复 GAPDecoder 接口"
    echo "     bash quick_fix_einsum.sh    # 修复 einsum 维度"
    echo "  3. 重新运行此验证脚本"
    exit 1
fi
