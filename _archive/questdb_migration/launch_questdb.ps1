# ──────────────────────────────────────────────────────────────────────────────
# launch_questdb.ps1 - Start QuestDB via the bundled-JRE native binary.
#
# Why we don't call questdb.exe: that wrapper insists on Administrator (it
# tries to install a Windows service). Invoking the bundled java.exe with
# the io.questdb module directly gives us the same server, no admin prompt.
# All state lands on D: per CLAUDE.md.
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$qdbDir   = Join-Path $ProjectRoot 'questdb\questdb-9.3.5-rt-windows-x86-64'
$java     = Join-Path $qdbDir 'bin\java.exe'
$dataRoot = Join-Path $ProjectRoot 'data\questdb_root'
$logFile  = Join-Path $ProjectRoot 'logs\questdb.log'

Write-Host "Starting QuestDB (bundled-JRE native binary)..." -ForegroundColor Cyan

if (-not (Test-Path $java)) {
    Write-Host "ERROR: $java not found." -ForegroundColor Red
    Write-Host "Run the install step first (download from https://questdb.io/download/)." -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Force -Path $dataRoot           | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $logFile) | Out-Null

# Already running on :9000? Skip - idempotent.
try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:9000/exec?query=SELECT%201' `
                           -UseBasicParsing -TimeoutSec 1.5
    if ($r.StatusCode -eq 200) {
        Write-Host "QuestDB already running on :9000 - skipping launch." -ForegroundColor Yellow
        exit 0
    }
} catch { }

$javaArgs = @('-m', 'io.questdb/io.questdb.ServerMain', '-d', $dataRoot)
$proc = Start-Process -FilePath $java -ArgumentList $javaArgs `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError ($logFile + '.err') `
    -WindowStyle Hidden `
    -PassThru
"$($proc.Id)" | Out-File -Encoding ascii -FilePath (Join-Path $ProjectRoot 'data\questdb.pid')

Write-Host ""
Write-Host "QuestDB started (PID $($proc.Id))." -ForegroundColor Green
Write-Host "  Web Console : http://localhost:9000" -ForegroundColor Cyan
Write-Host "  ILP (writes): localhost:9009"
Write-Host "  Postgres    : localhost:8812"
Write-Host "  Log         : $logFile"
Write-Host ""

# Wait briefly, then attempt schema creation.
Start-Sleep -Seconds 6
Write-Host "Creating tables (best-effort)..." -ForegroundColor Cyan
try {
    & "$ProjectRoot\venv\Scripts\python.exe" -m src.database.schema 2>&1 | Out-String | Write-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Tables ready." -ForegroundColor Green
    } else {
        Write-Host "Schema creation failed - QuestDB may still be starting; retry shortly." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Schema step skipped: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To ingest historical CSV.gz data:" -ForegroundColor Cyan
Write-Host "  python -m src.database.ingest_pipeline --timeframe 1m 1h 1d"
