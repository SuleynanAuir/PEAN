# PEAN 训练脚本
# 使用: .\run_train.ps1 [选项]

param(
    [int]$BatchSize = 32,
    [string]$Recognizer = "aster",
    [int]$SRB = 1,
    [switch]$PreTraining,
    [string]$Resume = ""
)

Write-Host "=== PEAN 训练脚本 ===" -ForegroundColor Cyan
Write-Host ""

# 激活conda环境
Write-Host "激活conda环境 'pean'..." -ForegroundColor Yellow
# conda activate pean  # PowerShell中需要手动激活

# 检查Python版本
Write-Host "当前Python版本:" -ForegroundColor Yellow
python --version
Write-Host ""

# 检查GPU
Write-Host "检查GPU状态..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}') "
Write-Host ""

# 检查训练数据
Write-Host "检查训练数据..." -ForegroundColor Yellow
$train1Exists = Test-Path "data\TextZoom\train1\data.mdb"
$train2Exists = Test-Path "data\TextZoom\train2\data.mdb"
Write-Host "  train1: $train1Exists"
Write-Host "  train2: $train2Exists"
Write-Host ""

if (-not $train1Exists -or -not $train2Exists) {
    Write-Host "警告: 训练数据不完整！请先下载train1和train2数据集。" -ForegroundColor Red
    Write-Host "参考 TRAIN_GUIDE.md 了解如何下载数据。" -ForegroundColor Yellow
    $continue = Read-Host "是否继续? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit
    }
}

# 构建训练命令
Write-Host "开始训练..." -ForegroundColor Green
Write-Host "参数配置:" -ForegroundColor Yellow
Write-Host "  Batch Size: $BatchSize"
Write-Host "  Recognizer: $Recognizer"
Write-Host "  SRB Blocks: $SRB"
Write-Host "  Pre-training: $PreTraining"
if ($Resume) {
    Write-Host "  Resume from: $Resume"
}
Write-Host ""

$cmd = "python main.py --batch_size=$BatchSize --mask --rec=`"$Recognizer`" --srb=$SRB"

if ($PreTraining) {
    $cmd += " --pre_training"
}

if ($Resume) {
    $cmd += " --resume=`"$Resume`""
}

Write-Host "执行命令: $cmd" -ForegroundColor Cyan
Write-Host ""
Write-Host "训练开始..." -ForegroundColor Green
Write-Host "按 Ctrl+C 可以停止训练" -ForegroundColor Yellow
Write-Host "==============================================================================================================`n"

Invoke-Expression $cmd

Write-Host "`n=============================================================================================================="
Write-Host "训练完成或已停止" -ForegroundColor Cyan
Write-Host ""
Write-Host "检查输出:" -ForegroundColor Yellow
Write-Host "  模型权重: .\ckpt\"
Write-Host "  训练日志: .\ckpt\log.csv"
