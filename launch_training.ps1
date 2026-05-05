$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'ML Training'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

# Pass any arguments from this script directly to the python script
& $python -u (Join-Path $root 'src\engine\train_all_models.py') $args 2>&1 | Out-String -Stream | Tee-Object -FilePath (Join-Path $root 'logs\training.log') -Append
