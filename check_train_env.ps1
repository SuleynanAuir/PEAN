# PEAN Training Environment Check Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PEAN Training Environment Check" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Check Python Environment
Write-Host "[1/6] Checking Python Environment..." -ForegroundColor Yellow
python --version
Write-Host ""

# 2. Check PyTorch and CUDA
Write-Host "[2/6] Checking PyTorch and CUDA..." -ForegroundColor Yellow
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  GPU Count: {torch.cuda.device_count()}'); print(f'  GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
Write-Host ""

# 3. Check Training Data
Write-Host "[3/6] Checking Training Data..." -ForegroundColor Yellow
$train1Exists = Test-Path "data\TextZoom\train1\data.mdb"
$train2Exists = Test-Path "data\TextZoom\train2\data.mdb"

if ($train1Exists) {
    $size1 = (Get-Item "data\TextZoom\train1\data.mdb").Length / 1GB
    Write-Host "  [OK] train1: Downloaded ($([math]::Round($size1, 2)) GB)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] train1: Not downloaded" -ForegroundColor Red
}

if ($train2Exists) {
    $size2 = (Get-Item "data\TextZoom\train2\data.mdb").Length / 1GB
    Write-Host "  [OK] train2: Downloaded ($([math]::Round($size2, 2)) GB)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] train2: Not downloaded" -ForegroundColor Red
}
Write-Host ""

# 4. Check Test Data
Write-Host "[4/6] Checking Test Data..." -ForegroundColor Yellow
$testEasy = Test-Path "data\TextZoom\test\easy\data.mdb"
$testMedium = Test-Path "data\TextZoom\test\medium\data.mdb"
$testHard = Test-Path "data\TextZoom\test\hard\data.mdb"

if($testEasy) { Write-Host "  [OK] Easy" -ForegroundColor Green } else { Write-Host "  [MISSING] Easy" -ForegroundColor Yellow }
if($testMedium) { Write-Host "  [OK] Medium" -ForegroundColor Green } else { Write-Host "  [MISSING] Medium" -ForegroundColor Yellow }
if($testHard) { Write-Host "  [OK] Hard" -ForegroundColor Green } else { Write-Host "  [MISSING] Hard" -ForegroundColor Yellow }
Write-Host ""

# 5. Check Recognizer Models
Write-Host "[5/6] Checking Recognizer Models..." -ForegroundColor Yellow
$aster = Test-Path "recognizers\aster.pth.tar"
$parseq = Test-Path "recognizers\parseq.pt"
$transformer = Test-Path "pretrain_transformer_stroke_decomposition.pth"

if($aster) { Write-Host "  [OK] ASTER" -ForegroundColor Green } else { Write-Host "  [MISSING] ASTER" -ForegroundColor Red }
if($parseq) { Write-Host "  [OK] PARSeq" -ForegroundColor Green } else { Write-Host "  [MISSING] PARSeq" -ForegroundColor Red }
if($transformer) { Write-Host "  [OK] Transformer (SFM)" -ForegroundColor Green } else { Write-Host "  [OPTIONAL] Transformer (SFM)" -ForegroundColor Yellow }
Write-Host ""

# 6. Check Config Files
Write-Host "[6/6] Checking Config Files..." -ForegroundColor Yellow
$configYaml = Test-Path "config\super_resolution.yaml"
$configJson = Test-Path "config\cfg_diff_prior.json"
$decompFile = Test-Path "english_decomposition.txt"

if($configYaml) { Write-Host "  [OK] super_resolution.yaml" -ForegroundColor Green } else { Write-Host "  [MISSING] super_resolution.yaml" -ForegroundColor Red }
if($configJson) { Write-Host "  [OK] cfg_diff_prior.json" -ForegroundColor Green } else { Write-Host "  [MISSING] cfg_diff_prior.json" -ForegroundColor Red }
if($decompFile) { Write-Host "  [OK] english_decomposition.txt" -ForegroundColor Green } else { Write-Host "  [MISSING] english_decomposition.txt" -ForegroundColor Red }
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Environment Status Summary" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$canTest = $testEasy -and $aster -and $parseq
$canTrain = $train1Exists -and $train2Exists -and $aster -and $parseq

if($canTest) { Write-Host "Testing Environment: READY" -ForegroundColor Green } else { Write-Host "Testing Environment: NOT READY" -ForegroundColor Red }
if($canTrain) { Write-Host "Training Environment: READY" -ForegroundColor Green } else { Write-Host "Training Environment: NOT READY" -ForegroundColor Red }
Write-Host ""

if (-not $canTrain) {
    Write-Host "Missing files for training:" -ForegroundColor Yellow
    if (-not $train1Exists) { Write-Host "   - TextZoom train1 dataset" -ForegroundColor Yellow }
    if (-not $train2Exists) { Write-Host "   - TextZoom train2 dataset" -ForegroundColor Yellow }
    if (-not $aster) { Write-Host "   - ASTER recognizer" -ForegroundColor Yellow }
    if (-not $parseq) { Write-Host "   - PARSeq recognizer" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Please refer to TRAIN_GUIDE.md for download instructions" -ForegroundColor Cyan
} else {
    Write-Host "All training files ready! You can start training now!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Start training with:" -ForegroundColor Cyan
    Write-Host "  .\run_train.ps1" -ForegroundColor White
    Write-Host "or" -ForegroundColor Cyan
    Write-Host "  python main.py --batch_size=32 --mask --rec=`"aster`" --srb=1" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================`n" -ForegroundColor Cyan
