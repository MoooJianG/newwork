# LDCSR 高斯查询方案 - 云服务器部署指南

> **适用场景**: 从 WSL 开发环境部署到云服务器（AutoDL、阿里云、AWS 等）
> **创建时间**: 2025-11-16
> **方案**: 高斯查询连续尺度超分（方案二）

---

## 📋 部署前检查清单（WSL 端）

### 1. 确认所有文件已创建

```bash
cd /home/millenn/newcode/LDCSR

# 检查核心文件
ls -lh modules/e2sr/gaussian_utils.py
ls -lh modules/e2sr/gaussian_query.py
ls -lh configs/first_stage_gaussian_query.yaml
ls -lh scripts/train_gaussian_query.sh
ls -lh scripts/test_gaussian_query.sh
ls -lh GAUSSIAN_QUERY_USAGE.md
```

**预期输出**:
```
-rw-r--r-- 1 user user  21K gaussian_utils.py
-rw-r--r-- 1 user user  19K gaussian_query.py
-rw-r--r-- 1 user user 4.5K first_stage_gaussian_query.yaml
-rwxr-xr-x 1 user user 6.2K train_gaussian_query.sh
-rwxr-xr-x 1 user user 7.5K test_gaussian_query.sh
-rw-r--r-- 1 user user  28K GAUSSIAN_QUERY_USAGE.md
```

### 2. 修复换行符问题

```bash
# 批量修复所有 shell 脚本
find scripts -name "*.sh" -type f -exec sed -i 's/\r$//' {} \;

# 验证
file scripts/train_gaussian_query.sh
# 应该看到: ASCII text (不应该有 CRLF)
```

### 3. 验证代码语法

```bash
# 测试 Python 导入
python -c "from modules.e2sr.gaussian_utils import make_coord"
python -c "from modules.e2sr.gaussian_query import GaussianQueryModule"

# 如果报错，检查 Python 版本
python --version  # 应该是 Python 3.10+
```

### 4. 清理不必要的文件

```bash
# 查看即将上传的文件
git status

# 确保以下目录/文件被忽略
ls -d logs/ 2>/dev/null && echo "⚠️  logs/ 应该被 .gitignore 忽略"
ls -d load/ 2>/dev/null && echo "⚠️  load/ 应该被 .gitignore 忽略"
ls -d results/ 2>/dev/null && echo "⚠️  results/ 应该被 .gitignore 忽略"

# 清理
git clean -fdx --dry-run  # 查看将被删除的文件
git clean -fdx             # 实际删除（谨慎！）
```

---

## 🚀 部署到云服务器

### 方案 A: 通过 Git 部署（推荐）

#### 步骤 1: 在 WSL 中提交代码

```bash
cd /home/millenn/newcode/LDCSR

# 添加所有新文件
git add modules/e2sr/gaussian_utils.py
git add modules/e2sr/gaussian_query.py
git add configs/first_stage_gaussian_query.yaml
git add scripts/train_gaussian_query.sh
git add scripts/test_gaussian_query.sh
git add GAUSSIAN_QUERY_USAGE.md
git add DEPLOYMENT_GUIDE.md

# 检查修改
git status

# 提交
git commit -m "feat: 实现高斯查询连续尺度超分（方案二）

- 添加 GaussianQueryModule 核心模块
- 集成到 GAPDecoder
- 支持任意尺度超分（包括分数尺度如 2.5x, 3.7x）
- 完整的训练和测试脚本
- 详细的使用文档
"

# 推送到远程仓库
git push origin main  # 或您的分支名
```

#### 步骤 2: 在云服务器拉取代码

```bash
# SSH 登录云服务器
ssh user@your-server-ip

# 克隆或拉取代码
cd /root/autodl-tmp/  # AutoDL 默认目录
git clone https://github.com/your-username/LDCSR.git
# 或者如果已克隆：
cd LDCSR
git pull origin main
```

### 方案 B: 通过 SCP/RSYNC 传输（快速）

```bash
# 从 WSL 传输到云服务器
rsync -avz --progress \
  --exclude='logs/' \
  --exclude='load/' \
  --exclude='results/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  /home/millenn/newcode/LDCSR/ \
  user@your-server-ip:/root/autodl-tmp/LDCSR/

# 或使用 SCP
scp -r /home/millenn/newcode/LDCSR user@your-server-ip:/root/autodl-tmp/
```

### 方案 C: 压缩传输（大文件）

```bash
# WSL 端
cd /home/millenn/newcode
tar -czf LDCSR_gaussian_query.tar.gz \
  --exclude='logs' \
  --exclude='load' \
  --exclude='results' \
  --exclude='__pycache__' \
  LDCSR/

# 传输
scp LDCSR_gaussian_query.tar.gz user@your-server-ip:/root/autodl-tmp/

# 云服务器端
cd /root/autodl-tmp
tar -xzf LDCSR_gaussian_query.tar.gz
```

---

## 🔧 云服务器环境配置

### 1. 安装依赖

```bash
# 登录云服务器
cd /root/autodl-tmp/LDCSR  # 或您的项目目录

# 创建 Conda 环境
conda create -n LDCSR python=3.10 -y
conda activate LDCSR

# 安装 PyTorch（根据您的 CUDA 版本）
# CUDA 11.8 (推荐)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirement.txt

# 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 2. 修复脚本权限

```bash
# 确保脚本可执行
chmod +x scripts/train_gaussian_query.sh
chmod +x scripts/test_gaussian_query.sh

# 再次修复换行符（如果需要）
find scripts -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### 3. 验证模块导入

```bash
# 测试核心模块
python << 'EOF'
from modules.e2sr.gaussian_utils import make_coord
from modules.e2sr.gaussian_query import GaussianQueryModule
print("✓ 高斯查询模块导入成功")

import torch
model = GaussianQueryModule(in_channels=64, out_channels=3)
print(f"✓ 模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
EOF
```

---

## 📦 准备数据集

### 选项 1: 下载真实数据集

```bash
# AID 数据集
cd /root/autodl-tmp/LDCSR

# 创建数据目录
mkdir -p dataset/RawAID

# 下载（根据实际情况）
# wget https://example.com/AID.zip
# 或使用云服务器的数据集市场

# 解压
unzip AID.zip -d dataset/RawAID/

# 准备数据
python data/prepare_split.py \
  --split_file AID_split.pkl \
  --data_path dataset/RawAID \
  --output_path load/AID_split

# 验证
ls load/AID_split/train/HR/ | wc -l
# 应该看到训练集图像数量
```

### 选项 2: 使用云盘数据（AutoDL 推荐）

```bash
# AutoDL 云盘路径
ln -s /root/autodl-fs/datasets/AID load/AID_split

# 验证
ls -l load/AID_split
```

---

## 🎯 开始训练

### 快速验证（调试模式）

```bash
# 激活环境
conda activate LDCSR

# 调试模式（5 epochs，10 batches）
bash scripts/train_gaussian_query.sh --gpus 0 --debug

# 观察输出
# - 应该看到 "✓ 使用高斯查询解码器"
# - 训练开始后会显示进度条
# - 5 epochs 后自动停止
```

### 完整训练

```bash
# 单卡训练
bash scripts/train_gaussian_query.sh --gpus 0

# 多卡训练（推荐）
bash scripts/train_gaussian_query.sh --gpus 0,1,2,3

# 后台训练（使用 nohup）
nohup bash scripts/train_gaussian_query.sh --gpus 0,1,2,3 > train.log 2>&1 &

# 查看日志
tail -f train.log

# 或使用 screen/tmux
screen -S ldcsr_train
bash scripts/train_gaussian_query.sh --gpus 0,1,2,3
# Ctrl+A, D 分离会话
# screen -r ldcsr_train 重新连接
```

### 监控训练

```bash
# TensorBoard（需要端口转发）
tensorboard --logdir logs/gaussian_query/ --port 6006 --host 0.0.0.0

# 在本地浏览器访问
# http://your-server-ip:6006

# 或使用 SSH 端口转发
# ssh -L 6006:localhost:6006 user@your-server-ip
# 然后访问 http://localhost:6006
```

---

## 📊 性能优化建议

### GPU 配置对应的参数

| GPU 配置 | batch_size | lr_img_sz | num_gaussians | 预估训练时间 (1000 epochs) |
|----------|-----------|-----------|---------------|------------------------|
| 1x V100 (16GB) | 1 | 32 | 50 | ~48 小时 |
| 1x A100 (40GB) | 2 | 48 | 100 | ~30 小时 |
| 4x V100 (16GB) | 2 | 48 | 100 | ~12 小时 |
| 4x A100 (40GB) | 4 | 48 | 200 | ~8 小时 |

### 内存不足时的降级方案

```bash
# 方案 1: 减小 batch_size
bash scripts/train_gaussian_query.sh --batch_size 1

# 方案 2: 修改配置文件
nano configs/first_stage_gaussian_query.yaml

# 修改以下参数:
# data.params.batch_size: 2 → 1
# data.params.train.params.lr_img_sz: 48 → 32
# model.params.decoder_config.params.num_gaussians: 100 → 50
# model.params.decoder_config.params.gaussian_unfold_row: 7 → 6

# 方案 3: 启用混合精度
# 在配置文件中添加:
# lightning:
#   trainer:
#     precision: 16
```

---

## 🧪 测试与评估

### 快速测试

```bash
# 找到最佳检查点
best_ckpt=$(ls -t logs/gaussian_query/*/checkpoints/epoch=*-best.ckpt 2>/dev/null | head -1)

# 如果没有 best，使用 last
if [ -z "$best_ckpt" ]; then
    best_ckpt=$(ls -t logs/gaussian_query/*/checkpoints/last.ckpt 2>/dev/null | head -1)
fi

echo "使用检查点: $best_ckpt"

# 测试
bash scripts/test_gaussian_query.sh --checkpoint $best_ckpt
```

### 任意尺度测试

```bash
# 测试分数尺度（方案二独有功能）
bash scripts/test_gaussian_query.sh \
  --checkpoint $best_ckpt \
  --scales 2.0,2.3,2.5,3.0,3.7,4.0,5.2,6.0,8.0
```

### 下载结果到本地

```bash
# 在 WSL 端执行
scp -r user@your-server-ip:/root/autodl-tmp/LDCSR/results/gaussian_query/ \
  /home/millenn/newcode/LDCSR/results/
```

---

## 🐛 常见问题排查

### Q1: 导入模块失败

```bash
# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 确认当前目录
pwd  # 应该在 /root/autodl-tmp/LDCSR

# 临时添加到路径
export PYTHONPATH=/root/autodl-tmp/LDCSR:$PYTHONPATH
```

### Q2: CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 卸载并重装 PyTorch
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 找不到数据集

```bash
# 检查路径
ls -l load/AID_split/train/HR/

# 如果不存在，检查配置
cat configs/first_stage_gaussian_query.yaml | grep datapath

# 修改为实际路径
nano configs/first_stage_gaussian_query.yaml
```

### Q4: 训练中断后恢复

```bash
# 找到最后的检查点
last_ckpt=$(ls -t logs/gaussian_query/*/checkpoints/last.ckpt | head -1)

# 恢复训练
bash scripts/train_gaussian_query.sh --resume $last_ckpt
```

---

## 📝 部署检查表

使用此检查表确保部署成功：

### WSL 端（部署前）

- [ ] 所有新文件已创建
- [ ] 换行符已修复（`sed -i 's/\r$//' scripts/*.sh`）
- [ ] Python 导入测试通过
- [ ] .gitignore 已更新
- [ ] 代码已提交到 Git（可选）

### 云服务器端（部署后）

- [ ] 代码已上传
- [ ] Conda 环境已创建
- [ ] 依赖已安装
- [ ] 脚本权限已设置
- [ ] 模块导入测试通过
- [ ] 数据集已准备
- [ ] GPU 可用（`nvidia-smi`）
- [ ] 调试模式训练成功

### 训练中

- [ ] TensorBoard 可访问
- [ ] 训练指标正常下降
- [ ] GPU 利用率 > 80%
- [ ] 定期保存检查点

### 训练后

- [ ] 找到最佳检查点
- [ ] 测试成功（整数尺度）
- [ ] 测试成功（任意尺度）
- [ ] 结果已下载到本地

---

## 🎓 进阶技巧

### 1. 使用 WandB 监控训练

```bash
pip install wandb

# 修改训练脚本或配置启用 WandB
# 或在训练前登录
wandb login
```

### 2. 自动重启训练（防止意外中断）

```bash
# 创建重启脚本
cat > auto_restart_train.sh << 'EOF'
#!/bin/bash
while true; do
    bash scripts/train_gaussian_query.sh --gpus 0,1,2,3
    if [ $? -eq 0 ]; then
        echo "训练正常结束"
        break
    else
        echo "训练中断，5秒后重启..."
        sleep 5
    fi
done
EOF

chmod +x auto_restart_train.sh
nohup ./auto_restart_train.sh > auto_train.log 2>&1 &
```

### 3. 定时保存检查点到云盘

```bash
# 创建备份脚本
cat > backup_checkpoints.sh << 'EOF'
#!/bin/bash
rsync -avz logs/gaussian_query/*/checkpoints/ /root/autodl-fs/backups/ldcsr_ckpt/
EOF

chmod +x backup_checkpoints.sh

# 添加到 crontab（每小时备份）
crontab -e
# 添加: 0 * * * * /root/autodl-tmp/LDCSR/backup_checkpoints.sh
```

---

## 📞 技术支持

- **问题排查**: 查看 `GAUSSIAN_QUERY_USAGE.md` 第 6 节（常见问题）
- **参数调优**: 查看 `GAUSSIAN_QUERY_USAGE.md` 第 5 节（参数调优指南）
- **代码细节**: 查看 `GAUSSIAN_INTEGRATION_PROPOSAL.md`

---

**最后更新**: 2025-11-16
**适用版本**: LDCSR v2.0 (高斯查询版)
