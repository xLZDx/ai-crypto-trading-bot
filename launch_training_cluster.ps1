# ──────────────────────────────────────────────────────────────────────────────
# launch_training_cluster.ps1  —  Start the Training Orchestrator (master node)
#
# Run this on the MASTER laptop (the one with training data).
# Worker laptops run worker.py pointing to this machine's IP.
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$ProjectRoot\logs\cluster.log"

$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Training Orchestrator — Master Node" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " This machine IP: $LocalIP"
Write-Host " Orchestrator:    http://${LocalIP}:7700"
Write-Host ""
Write-Host " To connect a worker laptop, run on that machine:" -ForegroundColor Yellow
Write-Host "   python -m src.training.distributed.worker --master http://${LocalIP}:7700" -ForegroundColor White
Write-Host ""
Write-Host " Dashboard cluster panel: http://localhost:5000 → Monitor tab" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot
python -m src.training.distributed.orchestrator --port 7700 2>&1 | Tee-Object -FilePath $LogFile
