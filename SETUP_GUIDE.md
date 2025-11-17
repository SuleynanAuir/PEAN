# PEAN 环境配置指南

本文档提供了在Windows上配置PEAN项目测试环境的完整步骤。

## 系统要求

- Windows 10/11
- NVIDIA GPU（支持CUDA 11.3）
- Miniconda 或 Anaconda

## 快速开始

### 1. 创建Conda环境

```powershell
conda create -n pean python=3.8 -y
conda activate pean
```

### 2. 配置pip镜像源（推荐，国内用户）

```powershell
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
```

### 3. 安装PyTorch（CUDA 11.3版本）

```powershell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 4. 降级pip（用于安装pytorch-lightning）

```powershell
python -m pip install pip==23.3.2
```

### 5. 安装其他依赖

使用阿里云镜像：
```powershell
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements_tested.txt
```

或使用官方源：
```powershell
pip install -r requirements_tested.txt
```

## 数据和模型下载

### 必需文件

1. **TextZoom测试数据集**
   - 下载: https://pan.baidu.com/s/1PYdNqo0GIeamkYHXJmRlDw (密码: kybq)
   - 解压到: `data/TextZoom/test/`

2. **ASTER识别器**
   - 下载: https://github.com/ayumiymk/aster.pytorch/releases/download/v1.0/demo.pth.tar
   - 保存为: `recognizers/aster.pth.tar`

3. **PARSeq识别器**
   - 自动下载（运行时会通过torch.hub下载）
   - 或手动保存到: `recognizers/parseq.pt`

4. **PEAN模型权重**
   - 百度网盘: https://pan.baidu.com/s/1Bu2WdoZ1gIfHz8JRujVq9w (密码: nr2n)
   - 或 Google Drive: https://drive.google.com/file/d/1kGhPN2wUCV12Cu4yX4WGgMer3U9sNNPu/view
   - 保存:
     - `ckpt/PEAN_final.pth`
     - `ckpt/TPEM_final.pth`

## 运行测试

激活环境并运行：

```powershell
conda activate pean
cd C:\Users\Aiur\PEAN

# 测试 Easy 子集
python main.py --batch_size=32 --mask --rec="aster" --srb=1 --resume="C:/Users/Aiur/PEAN/ckpt/PEAN_final.pth" --test --test_data_dir="C:/Users/Aiur/PEAN/data/TextZoom/test/easy"

# 测试 Medium 子集
python main.py --batch_size=32 --mask --rec="aster" --srb=1 --resume="C:/Users/Aiur/PEAN/ckpt/PEAN_final.pth" --test --test_data_dir="C:/Users/Aiur/PEAN/data/TextZoom/test/medium"

# 测试 Hard 子集
python main.py --batch_size=32 --mask --rec="aster" --srb=1 --resume="C:/Users/Aiur/PEAN/ckpt/PEAN_final.pth" --test --test_data_dir="C:/Users/Aiur/PEAN/data/TextZoom/test/hard"
```

或使用PowerShell脚本：
```powershell
.\run_test.ps1
```

## 预期结果

在TextZoom Easy测试集上：
- 识别准确率: ~81.5%
- PSNR: ~23.8 dB
- SSIM: ~0.866
- FPS: ~23.6

## 常见问题

### SSL连接错误
使用国内镜像源：
```powershell
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com <package>
```

### pytorch-lightning安装失败
需要降级pip版本：
```powershell
python -m pip install pip==23.3.2
```

### 缺少文件错误
确保已创建以下占位文件：
- `english_decomposition.txt` (已自动创建)

## 文件结构

```
PEAN/
├── ckpt/
│   ├── PEAN_final.pth
│   └── TPEM_final.pth
├── data/
│   └── TextZoom/
│       └── test/
│           ├── easy/
│           ├── medium/
│           └── hard/
├── recognizers/
│   ├── aster.pth.tar
│   └── parseq.pt
├── config/
│   ├── super_resolution.yaml
│   └── cfg_diff_prior.json
├── main.py
├── requirements_tested.txt
└── run_test.ps1
```

## 版本信息

- Python: 3.8.20
- PyTorch: 1.10.1+cu113
- CUDA: 11.3
- 测试日期: 2025-11-17
