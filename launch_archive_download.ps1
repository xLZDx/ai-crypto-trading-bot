$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'Binance Archive 1s Downloader'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Binance Archive Downloader - data.binance.vision"     -ForegroundColor Cyan
Write-Host "  Downloads full 1s history for all watchlist coins"    -ForegroundColor Cyan
Write-Host "  Files: data/raw/{SYMBOL}_spot_1s.csv.gz"             -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

& $python -u -m src.data_ingestion.binance_archive_downloader 2>&1 |
    Out-String -Stream |
    Tee-Object -FilePath (Join-Path $logDir 'archive_download.log') -Append

Write-Host ""
Write-Host "Archive download complete." -ForegroundColor Green
Read-Host "Press Enter to close"
