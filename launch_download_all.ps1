$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'Data Download Pipeline'
Set-Location $root
& $python -u (Join-Path $root 'src\data_ingestion\run_full_download.py') 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'download.log') -Append
Write-Host ""
Write-Host "ALL DOWNLOADS COMPLETE!" -ForegroundColor Green
Read-Host "Press Enter to close"
