$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trading Dashboard'
Set-Location $root
& $python (Join-Path $root 'src\dashboard\app.py') 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'dashboard.log') -Append
