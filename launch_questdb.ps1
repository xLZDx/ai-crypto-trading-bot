# ──────────────────────────────────────────────────────────────────────────────
# launch_questdb.ps1  —  Start QuestDB via Docker
#
# Prerequisites: Docker Desktop must be running.
#   Download: https://www.docker.com/products/docker-desktop/
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Starting QuestDB..." -ForegroundColor Cyan

# Check Docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker not found. Install Docker Desktop from https://www.docker.com" -ForegroundColor Red
    exit 1
}

# Pull latest image if not present
docker pull questdb/questdb:latest 2>$null

# Start (or restart existing container)
$existing = docker ps -a --filter "name=trading_questdb" --format "{{.Names}}" 2>$null
if ($existing -eq "trading_questdb") {
    Write-Host "Restarting existing container..." -ForegroundColor Yellow
    docker start trading_questdb
} else {
    Write-Host "Creating new QuestDB container..." -ForegroundColor Green
    docker run -d `
        --name trading_questdb `
        --restart unless-stopped `
        -p 9000:9000 `
        -p 9009:9009 `
        -p 8812:8812 `
        -v "$ProjectRoot\data\questdb:/root/.questdb" `
        -e QDB_TELEMETRY_ENABLED=false `
        -e QDB_SHARED_WORKER_COUNT=4 `
        -m 4g `
        questdb/questdb:latest
}

Write-Host ""
Write-Host "QuestDB started!" -ForegroundColor Green
Write-Host "  Web Console:  http://localhost:9000" -ForegroundColor Cyan
Write-Host "  ILP (writes): localhost:9009"
Write-Host ""
Write-Host "Creating tables..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

# Create schema
& python -m src.database.schema
if ($LASTEXITCODE -eq 0) {
    Write-Host "Tables ready." -ForegroundColor Green
} else {
    Write-Host "Schema creation failed — QuestDB may still be starting, retry in a few seconds." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To ingest historical CSV.gz data:" -ForegroundColor Cyan
Write-Host "  python -m src.database.ingest_pipeline --timeframe 1m 1h 1d"
Write-Host ""
Write-Host "To add a training worker on another laptop:" -ForegroundColor Cyan
Write-Host "  python -m src.training.distributed.worker --master http://$(hostname):7700"
