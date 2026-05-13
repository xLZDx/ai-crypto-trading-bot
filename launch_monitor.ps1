$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trading Monitor'
Set-Location $root
$logPath = Join-Path $logDir 'monitor.log'
# Remove stale log so new process gets a clean file handle (no lock contention)
Remove-Item -Path $logPath -Force -ErrorAction SilentlyContinue
& $python (Join-Path $root 'src\monitor\server.py') 2>&1 >> $logPath
