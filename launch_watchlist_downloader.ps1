$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'Watchlist Downloader Daemon'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

Write-Host "Starting Watchlist Downloader daemon (1s / 1m / 1M for all watchlist coins)..." -ForegroundColor Cyan

& $python -u -m src.data_ingestion.watchlist_downloader 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'watchlist_downloader.log') -Append

Write-Host ""
Write-Host "Watchlist Downloader exited." -ForegroundColor Yellow
Read-Host "Press Enter to close"
