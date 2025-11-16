#!/bin/bash
# ============================================================
# LDCSR 高斯查询方案 - 快速修复脚本
# ============================================================
#
# 修复问题: TypeError: GaussianQueryDecoder.forward() missing 1 required positional argument
#
# 使用方法: bash quick_fix.sh
#
# ============================================================

set -e

echo "=========================================="
echo " LDCSR 高斯查询方案 - 快速修复"
echo "=========================================="
echo ""

TARGET_FILE="modules/e2sr/s1_v6.py"

# 检查文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 错误: 文件不存在: $TARGET_FILE"
    echo "   请确保在 LDCSR 项目根目录运行此脚本"
    exit 1
fi

echo "✓ 找到目标文件: $TARGET_FILE"
echo ""

# 备份原文件
BACKUP_FILE="${TARGET_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "✓ 备份原文件: $BACKUP_FILE"
echo ""

# 检查是否已经修复
if grep -q "if self.use_gaussian_query:" "$TARGET_FILE"; then
    echo "✅ 文件已经修复，无需重复应用补丁"
    echo ""
    rm "$BACKUP_FILE"
    exit 0
fi

# 应用修复
echo "正在应用修复..."

# 使用 sed 替换代码
sed -i '/^        x_up = F\.interpolate(lr, out_size, mode="bicubic", align_corners=False)$/,/^        return out$/{
    s|^        x_up = F\.interpolate(lr, out_size, mode="bicubic", align_corners=False)$|        # ★ 根据是否使用高斯查询调用不同接口 ★\n        if self.use_gaussian_query:\n            # GaussianQueryDecoder 需要 (feat, lr, out_size)\n            out = self.upsampler(mid, lr, out_size)\n        else:\n            # SADNUpsampler 需要 (feat, out_size)\n            x_up = F.interpolate(lr, out_size, mode="bicubic", align_corners=False)|
    /^        out = self\.upsampler(mid, out_size) + x_up$/d
}' "$TARGET_FILE"

# 验证修复
if grep -q "if self.use_gaussian_query:" "$TARGET_FILE"; then
    echo "✅ 修复成功！"
    echo ""
    echo "修改内容:"
    echo "  - 添加了条件判断 use_gaussian_query"
    echo "  - GaussianQueryDecoder 使用 (feat, lr, out_size) 接口"
    echo "  - SADNUpsampler 保持原有接口"
    echo ""
    echo "下一步:"
    echo "  1. 重新运行训练: bash scripts/train_gaussian_query.sh"
    echo "  2. 如有问题，恢复备份: cp $BACKUP_FILE $TARGET_FILE"
else
    echo "❌ 自动修复失败，请手动修复"
    echo ""
    echo "手动修复步骤:"
    echo "  1. 编辑文件: nano $TARGET_FILE"
    echo "  2. 找到 GAPDecoder.forward() 方法的末尾"
    echo "  3. 查看 HOTFIX_decoder_forward.patch 获取详细修改"
    echo ""
    echo "备份文件: $BACKUP_FILE"
    exit 1
fi

echo "=========================================="
