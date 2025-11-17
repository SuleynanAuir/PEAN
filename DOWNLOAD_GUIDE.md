# PEAN 数据和模型下载指南

## 环境配置 ✅
- [x] Conda环境创建完成 (pean, Python 3.8.20)
- [x] PyTorch 1.10.1+cu113 安装完成
- [x] 所有依赖包安装完成

## 需要下载的内容

### 1. TextZoom 数据集 (必需)
**下载地址**: https://github.com/JasonBoy1/TextZoom

**操作步骤**:
1. 访问上述链接
2. 下载测试集 (test数据集包含 easy/medium/hard 三个子集)
3. 解压到 `C:\Users\Aiur\PEAN\data\TextZoom\`

**预期目录结构**:
```
C:\Users\Aiur\PEAN\data\TextZoom\
├── test\
│   ├── easy\
│   ├── medium\
│   └── hard\
```

### 2. 预训练识别器模型 (至少需要一个，推荐ASTER)

#### ASTER (推荐)
- **源码**: https://github.com/ayumiymk/aster.pytorch
- **模型下载**: 查看仓库的README获取预训练模型链接
- **保存路径**: `C:\Users\Aiur\PEAN\recognizers\aster.pth.tar`

#### CRNN (可选)
- **源码**: https://github.com/meijieru/crnn.pytorch
- **保存路径**: `C:\Users\Aiur\PEAN\recognizers\crnn.pth`

#### MORAN (可选)
- **源码**: https://github.com/Canjie-Luo/MORAN_v2
- **保存路径**: `C:\Users\Aiur\PEAN\recognizers\moran.pth`

#### PARSeq (可选)
- **源码**: https://github.com/baudm/parseq
- **保存路径**: `C:\Users\Aiur\PEAN\recognizers\parseq.pt`

### 3. PEAN 模型权重 (必需用于测试)

**下载地址** (选择一个):
- **百度网盘**: https://pan.baidu.com/s/1Bu2WdoZ1gIfHz8JRujVq9w (密码: nr2n)
- **Google Drive**: https://drive.google.com/file/d/1kGhPN2wUCV12Cu4yX4WGgMer3U9sNNPu/view?usp=sharing

**文件说明**:
- `PEAN_pretrained.pth` - 预训练模型 (用于 --pre_training --test)
- `PEAN_final.pth` - 完整模型主权重
- `PEAN_final.pth` - TPEM模块权重

**保存路径**:
```
C:\Users\Aiur\PEAN\ckpt\PEAN_final.pth
C:\Users\Aiur\PEAN\ckpt\TPEM_final.pth
```

### 4. SFM Loss的Transformer识别器 (可选，用于训练)
- **源码**: https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt
- 这个模型仅在训练时需要，测试时不需要

## 下载完成后的检查清单

运行以下PowerShell命令检查文件是否齐全:

```powershell
cd C:\Users\Aiur\PEAN
Write-Host "=== 检查数据集 ==="
Test-Path "data\TextZoom\test\easy" | ForEach-Object { Write-Host "TextZoom easy: $_" }
Test-Path "data\TextZoom\test\medium" | ForEach-Object { Write-Host "TextZoom medium: $_" }
Test-Path "data\TextZoom\test\hard" | ForEach-Object { Write-Host "TextZoom hard: $_" }

Write-Host "`n=== 检查识别器 ==="
Test-Path "recognizers\aster.pth.tar" | ForEach-Object { Write-Host "ASTER: $_" }

Write-Host "`n=== 检查模型权重 ==="
Test-Path "ckpt\PEAN_final.pth" | ForEach-Object { Write-Host "PEAN_final: $_" }
Test-Path "ckpt\TPEM_final.pth" | ForEach-Object { Write-Host "TPEM_final: $_" }
```

## 下载完成后的下一步

下载完成所有必需文件后，请告诉我，我将:
1. 更新配置文件中的路径
2. 运行测试脚本验证环境配置

## 最小测试配置

如果只是想快速测试，最少需要:
- ✅ TextZoom测试集的easy子集
- ✅ ASTER识别器
- ✅ PEAN_final.pth 和 TPEM_final.pth

有了这些就可以运行测试了！
