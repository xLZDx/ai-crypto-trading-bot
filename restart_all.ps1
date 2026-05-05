$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADER: POWERSHELL AUTO-SETUP"
Write-Host "   Root: $root"
Write-Host "==========================================" -ForegroundColor Cyan

# Step 0: ParquetClient store — file-based, no daemon. Just verifies the
# data directory + DuckDB import. (Was QuestDB Docker/native-binary launch
# before the Phase 1-5 migration; see commits 43db156..b64b733.)
Write-Host ""
Write-Host "[0/6] Verifying Parquet store (DuckDB)..." -ForegroundColor Yellow
$venvPy = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) { $venvPy = "python" }
$dbDir = Join-Path $root 'data\db'
if (-not (Test-Path $dbDir)) {
    New-Item -ItemType Directory -Force -Path $dbDir | Out-Null
}
& $venvPy -c "from src.database.parquet_client import get_client; c = get_client(); import sys; sys.exit(0 if c.is_available() else 1)" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Parquet store ready (DuckDB + $dbDir)." -ForegroundColor Green
} else {
    Write-Host "  WARNING: Parquet store unavailable. Run: pip install duckdb pyarrow" -ForegroundColor Red
}

Write-Host "  Running startup_recovery (archive gap fill)..."
$env:PYTHONIOENCODING = 'utf-8'
& $venvPy -m src.data_ingestion.startup_recovery --archive-only 2>&1 |
    Out-File -Append -FilePath (Join-Path $root 'logs\startup_recovery.log')
Write-Host "  Startup recovery complete." -ForegroundColor Green
Write-Host "[0/6] Parquet store ready." -ForegroundColor Green

# Step 1: Kill ONLY known managed processes (bot, dashboard, monitor, training).
#         Download processes (archive / watchlist) are NOT killed - they finish on their own.
Write-Host ""
Write-Host "[1/6] Terminating managed processes (bot/dashboard/monitor/training only)..." -ForegroundColor Yellow
$pidFile = Join-Path $root 'data\process_ids.json'
$killedPids = @()
if (Test-Path $pidFile) {
    try {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($key in @('bot','dash','monitor','training','realtime','orch','orderbook')) {
            $pidVal = $pids.$key
            if ($pidVal -and $pidVal -ne 0) {
                try {
                    Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                    $killedPids += $pidVal
                    Write-Host "  Stopped $key PID $pidVal"
                } catch {}
            }
        }
    } catch {
        Write-Host "  Could not parse PID file: $_" -ForegroundColor DarkYellow
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
# Fallback: kill python processes that have bot/dashboard/training scripts in their command line
Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null | ForEach-Object {
    $cmd = $_.CommandLine
    # NOTE: `train_all_models` was removed from this regex on 2026-05-05 so a
    # restart_all during a manual long-running retrain doesn't kill the
    # training process mid-pipeline. `launch_training` (the auto-scheduled
    # 10-min-after-boot trainer) is still killed.
    if ($cmd -match 'src\\main\.py|src/main\.py|launch_bot|src\\dashboard\\app|launch_dashboard|launch_training') {
        if ($_.ProcessId -notin $killedPids) {
            try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
            Write-Host "  Stopped stray process PID $($_.ProcessId): $($cmd.Substring(0,[Math]::Min(80,$cmd.Length)))"
        }
    }
}
Write-Host "[1/6] Managed processes terminated (downloads left running)." -ForegroundColor Green

Start-Sleep -Seconds 2

# Step 2: Virtual environment
Write-Host ""
Write-Host "[2/6] Setting up Virtual Environment..." -ForegroundColor Yellow
$venvPython = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating venv..." -ForegroundColor Magenta
    python -m venv (Join-Path $root 'venv')
    Write-Host "  Venv created."
} else {
    Write-Host "  Venv exists: $venvPython"
}
Write-Host "[2/6] Venv ready." -ForegroundColor Green

# Step 3: Install libraries
Write-Host ""
Write-Host "[3/6] Checking / installing libraries..." -ForegroundColor Yellow
$sp         = Join-Path $root 'venv\Lib\site-packages'
$hasFlask   = Test-Path (Join-Path $sp 'flask')
$hasCcxt    = Test-Path (Join-Path $sp 'ccxt')
$hasSklearn = Test-Path (Join-Path $sp 'sklearn')
$hasDarts   = Test-Path (Join-Path $sp 'darts')
Write-Host "  flask=$hasFlask  ccxt=$hasCcxt  sklearn=$hasSklearn  darts=$hasDarts"
if ($hasFlask -and $hasCcxt -and $hasSklearn -and $hasDarts) {
    Write-Host "[3/6] All core packages present - skipping install." -ForegroundColor Green
} else {
    Write-Host "  Installing packages (first run: 5-10 min)..." -ForegroundColor Magenta
    $pip = Join-Path $root 'venv\Scripts\pip.exe'
    & $venvPython -m pip install --upgrade pip --quiet
    & $pip install -r (Join-Path $root 'requirements.txt')
    Write-Host "[3/6] Libraries installed." -ForegroundColor Green
}

# Ensure dirs exist
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root 'data') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root 'data\raw\historical') | Out-Null
Write-Host "  Directories ready."

# Step 3.5: Redirect caches + set CPU/GPU env vars (shared setup_env.ps1)
Write-Host ""
Write-Host "[3.5/6] Applying D-drive cache redirect + CPU/GPU env vars..." -ForegroundColor Yellow
. (Join-Path $root 'setup_env.ps1')
Write-Host "[3.5/6] Environment configured." -ForegroundColor Green

# Helper: launch a .ps1 in the BACKGROUND (no visible window).
# -WindowStyle Hidden + -NoProfile keeps the desktop clean and starts faster.
# We drop -NoExit because the wrapper PS no longer needs to stay readable;
# logs are tailed via the dashboard's /api/monitor/logs/<component> endpoint.
function Start-Window {
    param([string]$Label, [string]$ScriptFile)
    Write-Host "  Launching $Label (hidden) ..."
    if (-not (Test-Path $ScriptFile)) {
        Write-Host "  WARNING: $ScriptFile not found" -ForegroundColor Red
        return $null
    }
    $argStr = '-NoProfile -ExecutionPolicy Bypass -File "' + $ScriptFile + '"'
    $proc = Start-Process powershell -ArgumentList $argStr -WindowStyle Hidden -PassThru
    Write-Host "  $Label started (PID $($proc.Id))"
    return $proc
}

# Step 0: Monitor server (dashboard)
Write-Host ""
Write-Host "[0/6] Starting Monitor server (http://127.0.0.1:5001)..." -ForegroundColor Yellow
$procMonitor = Start-Window -Label 'Monitor' -ScriptFile (Join-Path $root 'launch_monitor.ps1')
Start-Sleep -Seconds 3
Write-Host "[0/6] Monitor launched." -ForegroundColor Green

# Step 4: ML Training - deferred 10 minutes after restart so the bot stabilises first
Write-Host ""
Write-Host "[4/6] Scheduling ML Training to start in 10 minutes..." -ForegroundColor Yellow
$trainingScript = Join-Path $root 'launch_training.ps1'
$trainingDelay  = 600   # seconds
$procTraining = $null
if (Test-Path $trainingScript) {
    $argStr = "-NoProfile -ExecutionPolicy Bypass -Command `"Start-Sleep -Seconds $trainingDelay; & '$trainingScript'`""
    $procTraining = Start-Process powershell -ArgumentList $argStr -WindowStyle Hidden -PassThru
    Write-Host "  Training will start at $(( Get-Date ).AddSeconds($trainingDelay).ToString('HH:mm:ss')) (PID $($procTraining.Id), hidden)"
} else {
    Write-Host "  WARNING: launch_training.ps1 not found" -ForegroundColor Red
}
Write-Host "[4/6] Training scheduled." -ForegroundColor Green

# Step 5: Dashboard + Bot
Write-Host ""
Write-Host "[5/6] Launching Dashboard and Bot..." -ForegroundColor Yellow
$procDash = Start-Window -Label 'Dashboard' -ScriptFile (Join-Path $root 'launch_dashboard.ps1')
Start-Sleep -Seconds 2
$procBot  = Start-Window -Label 'Bot'       -ScriptFile (Join-Path $root 'launch_bot.ps1')
Start-Sleep -Seconds 2
Write-Host "[5/6] Dashboard and Bot launched." -ForegroundColor Green

# Step 5.5: Watchlist Downloader Daemon (only start if not already running)
Write-Host ""
Write-Host "[5.5/6] Checking Watchlist Downloader Daemon..." -ForegroundColor Yellow
$wdRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'watchlist_downloader' }
if ($wdRunning) {
    Write-Host "  Watchlist Downloader already running (PID $($wdRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procWatchlist = Get-Process -Id $wdRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $procWatchlist = Start-Window -Label 'WatchlistDownloader' -ScriptFile (Join-Path $root 'launch_watchlist_downloader.ps1')
    Start-Sleep -Seconds 2
}
Write-Host "[5.5/6] Watchlist Downloader ready." -ForegroundColor Green

# Step 5.6: FastAPI Control Plane (Phase 13) - :8100 health/status/control
Write-Host ""
Write-Host "[5.6/6] Starting FastAPI Control Plane (:8100)..." -ForegroundColor Yellow
$fapiRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'src\.server\.control_plane' }
if ($fapiRunning) {
    Write-Host "  FastAPI already running (PID $($fapiRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procFastapi = Get-Process -Id $fapiRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $procFastapi = Start-Window -Label 'FastAPI' -ScriptFile (Join-Path $root 'launch_fastapi.ps1')
    Start-Sleep -Seconds 2
}
Write-Host "[5.6/6] FastAPI Control Plane ready." -ForegroundColor Green

# Step 5.7: Realtime DB Writer (Phase 7) - Binance WS -> QuestDB
Write-Host ""
Write-Host "[5.7/6] Starting Realtime DB Writer (Binance WS -> Parquet)..." -ForegroundColor Yellow
$rtRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'realtime_db_writer' }
if ($rtRunning) {
    Write-Host "  Realtime DB Writer already running (PID $($rtRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procRealtime = Get-Process -Id $rtRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $rtArgStr = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$venvPython' -m src.data_ingestion.realtime_db_writer 2>&1 | Tee-Object -FilePath '$root\logs\realtime_db.log' -Append`""
    $procRealtime = Start-Process powershell -ArgumentList $rtArgStr -WindowStyle Hidden -PassThru
    Write-Host "  Realtime DB Writer started (PID $($procRealtime.Id), hidden)"
}
Write-Host "[5.7/6] Realtime DB Writer ready." -ForegroundColor Green

# Step 5.8: Data Governance Orchestrator (Phase 8) - multi-source ingest
Write-Host ""
Write-Host "[5.8/6] Starting Data Governance Orchestrator (multi-source feeds)..." -ForegroundColor Yellow
$orchRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'data_governance.orchestrator' }
if ($orchRunning) {
    Write-Host "  Orchestrator already running (PID $($orchRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procOrch = Get-Process -Id $orchRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $orchArgStr = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$venvPython' -m src.data_governance.orchestrator 2>&1 | Tee-Object -FilePath '$root\logs\data_orchestrator.log' -Append`""
    $procOrch = Start-Process powershell -ArgumentList $orchArgStr -WindowStyle Hidden -PassThru
    Write-Host "  Data Orchestrator started (PID $($procOrch.Id), hidden)"
}
Write-Host "[5.8/6] Data Orchestrator ready." -ForegroundColor Green

# Step 5.9: L2 Order Book Collector (Phase 1) - feeds OFT model + ZeroMQ data plane
# Set $env:OB_COLLECTOR_DISABLED='1' to skip (e.g. on metered connections).
Write-Host ""
Write-Host "[5.9/6] Starting L2 Order Book Collector..." -ForegroundColor Yellow
$obSymbols = if ($env:OB_COLLECTOR_SYMBOLS) { $env:OB_COLLECTOR_SYMBOLS } else { 'BTC/USDT,ETH/USDT,SOL/USDT' }
if ($env:OB_COLLECTOR_DISABLED -eq '1') {
    Write-Host "  Skipped (OB_COLLECTOR_DISABLED=1)." -ForegroundColor DarkCyan
    $procOrderbook = $null
} else {
    $obRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
        Where-Object { $_.CommandLine -match 'orderbook_collector' }
    if ($obRunning) {
        Write-Host "  Orderbook Collector already running (PID $($obRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
        $procOrderbook = Get-Process -Id $obRunning.ProcessId -ErrorAction SilentlyContinue
    } else {
        $obArgStr = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$venvPython' -m src.data_ingestion.orderbook_collector --symbols '$obSymbols' --depth 20 --speed 100ms 2>&1 | Tee-Object -FilePath '$root\logs\orderbook_collector.log' -Append`""
        $procOrderbook = Start-Process powershell -ArgumentList $obArgStr -WindowStyle Hidden -PassThru
        Write-Host "  Orderbook Collector started (PID $($procOrderbook.Id), symbols: $obSymbols)"
    }
}
Write-Host "[5.9/6] Orderbook Collector ready." -ForegroundColor Green

# Step 6: Save PIDs
Write-Host ""
Write-Host "[6/6] Saving process IDs..." -ForegroundColor Yellow
$monId       = if ($procMonitor)   { $procMonitor.Id   } else { 0 }
$dashId      = if ($procDash)      { $procDash.Id      } else { 0 }
$botId       = if ($procBot)       { $procBot.Id       } else { 0 }
$watchlistId = if ($procWatchlist) { $procWatchlist.Id } else { 0 }
$trainingId  = if ($procTraining)  { $procTraining.Id  } else { 0 }
$realtimeId  = if ($procRealtime)  { $procRealtime.Id  } else { 0 }
$orchId      = if ($procOrch)      { $procOrch.Id      } else { 0 }
$fastapiId   = if ($procFastapi)   { $procFastapi.Id   } else { 0 }
$obId        = if ($procOrderbook) { $procOrderbook.Id } else { 0 }
$pidData = @{ bot = $botId; dash = $dashId; monitor = $monId; watchlist = $watchlistId; training = $trainingId; realtime = $realtimeId; orch = $orchId; fastapi = $fastapiId; orderbook = $obId; mcp = 0 }
$pidData | ConvertTo-Json | Set-Content (Join-Path $root 'data\process_ids.json')
Write-Host "[6/6] PIDs saved: monitor=$monId  dash=$dashId  bot=$botId  watchlist=$watchlistId  training=$trainingId  realtime=$realtimeId  orch=$orchId  fastapi=$fastapiId  orderbook=$obId" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   ALL PROCESSES STARTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "   Monitor   -> http://127.0.0.1:5001"   -ForegroundColor Cyan
Write-Host "   Dashboard -> http://127.0.0.1:5000"   -ForegroundColor Cyan
Write-Host "   Training starts in 10 minutes."        -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "This window stays open for your reference." -ForegroundColor White
Read-Host "Press Enter to close"
