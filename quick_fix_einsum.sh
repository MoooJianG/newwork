#!/bin/bash
# ============================================================
# LDCSR 高斯查询方案 - einsum 维度快速修复脚本
# ============================================================
#
# 修复问题: RuntimeError: einsum() dimension mismatch
#
# 使用方法: bash quick_fix_einsum.sh
#
# ============================================================

set -e

echo "=========================================="
echo " LDCSR 高斯查询 - einsum 维度修复"
echo "=========================================="
echo ""

TARGET_FILE="modules/e2sr/gaussian_query.py"

# 检查文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 错误: 文件不存在: $TARGET_FILE"
    echo "   请确保在 LDCSR 项目根目录运行此脚本"
    exit 1
fi

echo "✓ 找到目标文件: $TARGET_FILE"
echo ""

# 备份原文件
BACKUP_FILE="${TARGET_FILE}.backup_einsum_$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "✓ 备份原文件: $BACKUP_FILE"
echo ""

# 检查是否已经修复
if grep -q "xy_flat = xy.reshape(num_LR_points, -1, 2)" "$TARGET_FILE"; then
    echo "✅ 文件已经修复，无需重复应用补丁"
    echo ""
    rm "$BACKUP_FILE"
    exit 0
fi

# 应用修复
echo "正在应用修复..."

# 使用 Python 进行精确替换（避免 sed 的转义问题）
python3 << 'EOF'
import re

target_file = "modules/e2sr/gaussian_query.py"

with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 原始代码模式
old_pattern = r'''        # 5\.3 计算高斯值
        z = torch\.einsum\('b\.\.\.i,bij,b\.\.\.j->b\.\.\.', xy, -0\.5 \* inv_covariance, xy\)'''

# 新代码
new_code = '''        # 5.3 计算高斯值（马氏距离）
        # xy: [num_LR_points, k, k, 2]
        # inv_covariance: [num_LR_points, 2, 2]
        # 需要计算: z = -0.5 * xy @ inv_covariance @ xy^T

        # 重塑 xy 为 [num_LR_points, k*k, 2]
        xy_flat = xy.reshape(num_LR_points, -1, 2)

        # 计算 xy @ inv_covariance: [num_LR_points, k*k, 2]
        temp = torch.matmul(xy_flat, -0.5 * inv_covariance)

        # 计算内积: [num_LR_points, k*k]
        z_flat = (temp * xy_flat).sum(dim=-1)

        # 重塑回 [num_LR_points, k, k]
        z = z_flat.reshape(num_LR_points, self.kernel_size, self.kernel_size)'''

# 替换
new_content = re.sub(old_pattern, new_code, content)

if new_content == content:
    print("警告: 未找到匹配的代码模式，尝试备用方案...")
    # 备用方案：直接查找 einsum 行
    if "torch.einsum('b...i,bij,b...j->b...'" in content:
        # 找到并替换该行及其上一行
        lines = content.split('\n')
        for i in range(len(lines)):
            if "torch.einsum('b...i,bij,b...j->b...'" in lines[i]:
                # 替换注释和 einsum 行
                lines[i-1] = "        # 5.3 计算高斯值（马氏距离）"
                lines[i] = "        # xy: [num_LR_points, k, k, 2]"
                # 插入新代码
                insert_lines = [
                    "        # inv_covariance: [num_LR_points, 2, 2]",
                    "        # 需要计算: z = -0.5 * xy @ inv_covariance @ xy^T",
                    "",
                    "        # 重塑 xy 为 [num_LR_points, k*k, 2]",
                    "        xy_flat = xy.reshape(num_LR_points, -1, 2)",
                    "",
                    "        # 计算 xy @ inv_covariance: [num_LR_points, k*k, 2]",
                    "        temp = torch.matmul(xy_flat, -0.5 * inv_covariance)",
                    "",
                    "        # 计算内积: [num_LR_points, k*k]",
                    "        z_flat = (temp * xy_flat).sum(dim=-1)",
                    "",
                    "        # 重塑回 [num_LR_points, k, k]",
                    "        z = z_flat.reshape(num_LR_points, self.kernel_size, self.kernel_size)",
                ]
                for idx, line in enumerate(insert_lines):
                    lines.insert(i + 1 + idx, line)
                break
        new_content = '\n'.join(lines)
    else:
        print("错误: 未找到目标代码")
        exit(1)

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ 代码替换成功")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Python 脚本执行失败"
    echo "恢复备份: cp $BACKUP_FILE $TARGET_FILE"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

# 验证修复
if grep -q "xy_flat = xy.reshape(num_LR_points, -1, 2)" "$TARGET_FILE"; then
    echo "✅ 修复成功！"
    echo ""
    echo "修改内容:"
    echo "  - 移除了有问题的 einsum 操作"
    echo "  - 使用显式矩阵乘法计算马氏距离"
    echo "  - 添加了详细的维度注释"
    echo ""
    echo "下一步:"
    echo "  1. 重新运行训练: bash scripts/train_gaussian_query.sh"
    echo "  2. 如有问题，恢复备份: cp $BACKUP_FILE $TARGET_FILE"
else
    echo "❌ 自动修复失败，请手动修复"
    echo ""
    echo "手动修复步骤:"
    echo "  1. 编辑文件: nano $TARGET_FILE"
    echo "  2. 找到第 191-196 行（einsum 相关代码）"
    echo "  3. 查看 HOTFIX_einsum_dims.patch 获取详细修改"
    echo ""
    echo "备份文件: $BACKUP_FILE"
    exit 1
fi

echo "=========================================="
