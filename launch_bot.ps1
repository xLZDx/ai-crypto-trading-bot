$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trading Bot'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

& $python (Join-Path $root 'src\main.py') 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'bot.log') -Append
