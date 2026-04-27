Set-Location $PSScriptRoot
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   INSTALLING CUDA TORCH FOR RTX 3080 Ti   " -ForegroundColor Cyan
Write-Host "   (requires CUDA 12.x driver)             " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$pip = ".\venv\Scripts\pip.exe"

Write-Host "`nUninstalling CPU-only torch..." -ForegroundColor Yellow
& $pip uninstall torch torchvision torchaudio -y 2>&1

Write-Host "`nInstalling torch with CUDA 12.4 support..." -ForegroundColor Yellow
Write-Host "(Downloading ~2.5 GB — this will take a few minutes)`n" -ForegroundColor Gray
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "`nVerifying CUDA availability..." -ForegroundColor Yellow
$result = & ".\venv\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
Write-Host $result -ForegroundColor Cyan

if ($result -match "CUDA available: True") {
    Write-Host "`nCUDA torch installed successfully! TFT training will now use your RTX 3080 Ti." -ForegroundColor Green
} else {
    Write-Host "`nWARNING: CUDA not detected. Check that NVIDIA drivers support CUDA 12.4+." -ForegroundColor Red
    Write-Host "Run 'nvidia-smi' in a terminal to check your driver version." -ForegroundColor Yellow
    Write-Host "If driver is older than 520.x, try cu118 instead of cu124:" -ForegroundColor Yellow
    Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Gray
}
