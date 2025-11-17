# PEAN 测试脚本
# 使用: .\run_test.ps1

Write-Host "=== PEAN 测试脚本 ===" -ForegroundColor Cyan
Write-Host ""

# 激活conda环境
Write-Host "激活conda环境 'pean'..." -ForegroundColor Yellow
conda activate pean

# 检查是否在正确的环境中
Write-Host "当前Python版本:" -ForegroundColor Yellow
python --version
Write-Host ""

# 测试 easy 子集
Write-Host "开始测试 TextZoom Easy 子集..." -ForegroundColor Green
python main.py `
    --batch_size=32 `
    --mask `
    --rec="aster" `
    --srb=1 `
    --resume="C:/Users/Aiur/PEAN/ckpt/PEAN_final.pth" `
    --test `
    --test_data_dir="C:/Users/Aiur/PEAN/data/TextZoom/test/easy"

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan
