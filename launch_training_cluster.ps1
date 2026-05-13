# ──────────────────────────────────────────────────────────────────────────────
# launch_training_cluster.ps1  -  Start the Training Orchestrator (master node)
#
# Run this on the MASTER laptop (the one with training data).
# Worker laptops run worker.py pointing to this machine's IP.
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$ProjectRoot\logs\cluster.log"

# Prefer 192.168.0.x LAN interface; fall back to any non-loopback IPv4
$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -like "192.168.0.*" } |
    Select-Object -First 1).IPAddress

if (-not $LocalIP) {
    $LocalIP = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } |
        Select-Object -First 1).IPAddress
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Training Orchestrator - Master Node" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " This machine IP: $LocalIP"
Write-Host " Orchestrator:    http://${LocalIP}:7700"
Write-Host ""
Write-Host " To connect a worker laptop, run on that machine:" -ForegroundColor Yellow
Write-Host "   python -m src.training.distributed.worker --master http://${LocalIP}:7700" -ForegroundColor White
Write-Host ""
Write-Host " Dashboard cluster panel: http://localhost:5000 -> Monitor tab" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot
# 2026-05-12 Phase A2: the orchestrator defaults to bind 127.0.0.1
# now. This launcher is for cluster mode (master + remote workers),
# so we explicitly pass --host 0.0.0.0 to accept inbound from any
# LAN worker. If you want single-machine only, drop the --host flag
# or run "python -m src.training.distributed.orchestrator" directly.
python -m src.training.distributed.orchestrator --port 7700 --host 0.0.0.0 2>&1 | Tee-Object -FilePath $LogFile
