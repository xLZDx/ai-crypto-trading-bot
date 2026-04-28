$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'ML Training'
Set-Location $root
& $python -u (Join-Path $root 'src\engine\train_all_models.py') 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $logDir 'training.log') -Append
