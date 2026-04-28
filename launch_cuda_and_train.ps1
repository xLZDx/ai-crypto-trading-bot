$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$pip    = Join-Path $root 'venv\Scripts\pip.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'CUDA Install + Full Training'
Set-Location $root

function Test-TorchCuda {
    $r = & $python -c "import torch; print(torch.cuda.is_available())" 2>&1
    return ($r -join '') -match 'True'
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   STEP 1: INSTALL CUDA TORCH (RTX 3080 Ti)"
Write-Host "   Python 3.14 — trying multiple CUDA builds"
Write-Host "==========================================" -ForegroundColor Cyan

# Uninstall whatever is there
Write-Host "`nUninstalling existing torch..." -ForegroundColor Yellow
& $pip uninstall torch torchvision torchaudio -y 2>&1 | Out-String -Stream

$installed = $false

# Attempt 1: cu128 (newest — most likely to have Python 3.14 wheels)
if (-not $installed) {
    Write-Host "`nAttempt 1/4: CUDA 12.8 wheel..." -ForegroundColor Yellow
    & $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | Out-String -Stream
    if (Test-TorchCuda) { $installed = $true; Write-Host "SUCCESS: cu128!" -ForegroundColor Green }
}

# Attempt 2: cu126
if (-not $installed) {
    Write-Host "`nAttempt 2/4: CUDA 12.6 wheel..." -ForegroundColor Yellow
    & $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 2>&1 | Out-String -Stream
    if (Test-TorchCuda) { $installed = $true; Write-Host "SUCCESS: cu126!" -ForegroundColor Green }
}

# Attempt 3: cu124
if (-not $installed) {
    Write-Host "`nAttempt 3/4: CUDA 12.4 wheel..." -ForegroundColor Yellow
    & $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | Out-String -Stream
    if (Test-TorchCuda) { $installed = $true; Write-Host "SUCCESS: cu124!" -ForegroundColor Green }
}

# Attempt 4: nightly cu128 (pre-release, broadest Python support)
if (-not $installed) {
    Write-Host "`nAttempt 4/4: nightly cu128 (pre-release)..." -ForegroundColor Yellow
    & $pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | Out-String -Stream
    if (Test-TorchCuda) { $installed = $true; Write-Host "SUCCESS: nightly cu128!" -ForegroundColor Green }
}

if (-not $installed) {
    Write-Host "`nWARNING: No CUDA torch found for Python 3.14." -ForegroundColor Red
    Write-Host "Reinstalling CPU torch so training can continue..." -ForegroundColor Yellow
    & $pip install torch torchvision 2>&1 | Out-String -Stream
    Write-Host "CPU torch reinstalled. TFT will train on CPU (slower)." -ForegroundColor Yellow
} else {
    Write-Host "`nGPU verify:" -ForegroundColor Green
    & $python -u -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>&1 | Out-String -Stream
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "   STEP 2: FULL TRAINING PIPELINE"
Write-Host "   (data download + 6 models)"
Write-Host "==========================================" -ForegroundColor Cyan

& $python -u (Join-Path $root 'src\engine\train_all_models.py') 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'training.log') -Append

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "   TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Read-Host "Press Enter to close"
