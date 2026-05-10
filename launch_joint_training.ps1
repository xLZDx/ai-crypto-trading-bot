$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$venvPython = Join-Path $root 'venv\Scripts\python.exe'
$logFile    = Join-Path $root 'logs\joint_training.log'
New-Item -ItemType Directory -Force -Path (Split-Path $logFile) | Out-Null

$env:PYTHONIOENCODING = 'utf-8'
Write-Host "[joint-train] Starting OFT + SAC joint training..." -ForegroundColor Yellow
Write-Host "[joint-train] Log: $logFile"
& $venvPython -m src.training.joint_oft_rl --symbol "BTC/USDT" --tf 1m --epochs 5 --episodes 30 2>&1 |
    Tee-Object -FilePath $logFile -Append
Write-Host "[joint-train] DONE" -ForegroundColor Green
