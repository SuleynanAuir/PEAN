# PEAN 训练数据下载和配置指南

## 📦 训练所需数据

### 1. TextZoom 训练数据集（必需）

**下载地址**（选择一个）:
- **百度网盘**: https://pan.baidu.com/s/1PYdNqo0GIeamkYHXJmRlDw
  - 提取码: `kybq`
- **Google Drive**: https://drive.google.com/drive/folders/1WRVy-fC_KrembPkaI68uqQ9wyaptibMh?usp=sharing

**需要下载的文件**:
- `train1/` - 训练集第1部分（LMDB格式）
- `train2/` - 训练集第2部分（LMDB格式）

**解压后目录结构**:
```
C:\Users\Aiur\PEAN\data\TextZoom\
├── train1\
│   ├── data.mdb
│   └── lock.mdb
├── train2\
│   ├── data.mdb
│   └── lock.mdb
└── test\
    ├── easy\
    ├── medium\
    └── hard\
```

### 2. SFM Loss 的 Transformer 识别器（训练时需要）

**来源**: FudanOCR Text-Gestalt 项目
- **GitHub**: https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt

**下载步骤**:
1. 访问上述链接
2. 查找预训练模型下载链接
3. 下载模型文件
4. 保存为: `C:\Users\Aiur\PEAN\pretrain_transformer_stroke_decomposition.pth`

**注意**: 
- 这个模型用于计算 Stroke Focus Loss（SFM Loss）
- 仅在训练时需要，测试时不需要
- 如果找不到预训练模型，可以先跳过此步骤，训练时会给出提示

### 3. 其他已下载的文件（训练也需要）

确保以下文件已经下载：
- ✅ `recognizers/aster.pth.tar` - ASTER识别器
- ✅ `recognizers/parseq.pt` - PARSeq识别器
- ✅ `english_decomposition.txt` - 笔画分解文件

## 🔧 配置文件修改

训练前需要确认配置文件已正确设置：

### 1. super_resolution.yaml

已配置的训练路径（确认即可）:
```yaml
TRAIN:
  train_data_dir: [
    'C:/Users/Aiur/PEAN/data/TextZoom/train1',
    'C:/Users/Aiur/PEAN/data/TextZoom/train2'
  ]
  batch_size: 512  # 可根据GPU显存调整
  ...
```

### 2. cfg_diff_prior.json

检查TPEM模型保存路径:
```json
"path": { 
    "checkpoint": "./ckpt",  # TPEM权重保存目录
    ...
}
```

## 📝 训练命令

### 方式1: 直接训练完整模型（推荐）

```powershell
conda activate pean
cd C:\Users\Aiur\PEAN

# 从头开始训练
python main.py --batch_size=32 --mask --rec="aster" --srb=1
```

**参数说明**:
- `--batch_size=32`: 批次大小（可根据显存调整）
- `--mask`: 使用mask机制
- `--rec="aster"`: 使用ASTER识别器
- `--srb=1`: SRB模块数量

### 方式2: 预训练 + 微调（两阶段训练）

**第1步 - 预训练**:
```powershell
python main.py --batch_size=32 --mask --rec="aster" --srb=1 --pre_training
```

预训练模型会保存在 `./ckpt/` 目录下

**第2步 - 微调**:
```powershell
# 假设预训练模型保存为 checkpoint.pth
python main.py --batch_size=32 --mask --rec="aster" --srb=1 --resume="./ckpt/checkpoint.pth"
```

### 调整训练参数

**降低显存占用**:
```powershell
python main.py --batch_size=16 --mask --rec="aster" --srb=1  # 减小batch_size
```

**修改其他参数**:
```powershell
python main.py --batch_size=32 --mask --rec="aster" --srb=1 \
  --hd_u=32 \           # hidden units
  --srb=5 \             # SRB blocks数量
  --dropout=0.1         # dropout率
```

## 📊 训练监控

### 1. 日志文件
训练日志会保存在:
- `./ckpt/log.csv` - 包含每个epoch的accuracy、PSNR、SSIM等指标

### 2. TensorBoard（如果启用）
```powershell
tensorboard --logdir=./tb_logger
```

### 3. Wandb（如果配置）
训练会自动记录到Weights & Biases平台

## 🚨 常见问题

### 1. CUDA Out of Memory
**解决方案**:
- 减小 `batch_size`
- 减少 `srb` 数量
- 使用更小的图像尺寸

### 2. 找不到 pretrain_transformer_stroke_decomposition.pth
**解决方案**:
- 已修改代码，如果文件不存在会跳过加载
- 可以继续训练，但可能影响SFM Loss的效果

### 3. 训练速度慢
**建议**:
- 确保使用GPU: 检查 `torch.cuda.is_available()`
- 减少验证频率: 修改 `valInterval` 参数
- 使用更少的worker: 修改 `workers` 参数

## 📁 训练输出文件

训练完成后会生成:
```
ckpt/
├── checkpoint_epoch_xxx.pth     # 每个epoch的PEAN权重
├── TPEM_epoch_xxx.pth          # 每个epoch的TPEM权重  
├── best_model.pth              # 最佳PEAN模型
├── best_TPEM.pth               # 最佳TPEM模型
└── log.csv                     # 训练日志
```

## ✅ 开始训练检查清单

在开始训练前，确认:
- [ ] TextZoom train1 和 train2 数据已下载并放在正确位置
- [ ] ASTER 识别器已下载
- [ ] PARSeq 识别器已下载
- [ ] 配置文件路径已正确设置
- [ ] GPU 可用且驱动正常
- [ ] 有足够的磁盘空间保存模型（建议至少50GB）
- [ ] conda 环境已激活

## 🎯 快速开始命令

```powershell
# 1. 激活环境
conda activate pean

# 2. 进入项目目录
cd C:\Users\Aiur\PEAN

# 3. 开始训练（小batch size测试）
python main.py --batch_size=16 --mask --rec="aster" --srb=1

# 4. 监控训练（另开一个终端）
Get-Content .\ckpt\log.csv -Wait
```

完成数据下载后，请告诉我，我会帮您运行训练！
