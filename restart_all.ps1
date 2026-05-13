$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADER: POWERSHELL AUTO-SETUP"
Write-Host "   Root: $root"
Write-Host "==========================================" -ForegroundColor Cyan

# Start-Detached -- create a fully detached process via WMI Win32_Process.Create.
# Why: Start-Process powershell -WindowStyle Hidden -PassThru spawns a child
# that shares the parent's console session. When the parent (e.g. the Bash
# shell that ran this script, or a closed Windows Terminal tab) ends, all
# children sharing the console get CTRL_CLOSE_EVENT and die. This was the
# root cause of dashboards/bots crashing silently within minutes of every
# restart_all (logs/dashboard.log just stops mid-stream, no traceback).
#
# WMI Win32_Process.Create runs the new process in the context of the WMI
# service -- it has NO parent in our shell tree, so console-group death
# doesn't reach it. Returns the new PID (or $null on failure).
function Start-Detached {
    param(
        [Parameter(Mandatory=$true)][string]$CommandLine,
        [string]$LogFile = $null
    )
    if ($LogFile) {
        # Funnel stdout+stderr to the log file at the OS file-handle level
        # (no PowerShell pipeline). cmd /S /C "..." parses the inner string
        # as the command -- /S preserves the outer quote pair so paths with
        # spaces (D:\test 2\...) survive intact. Without /S, cmd's default
        # rule strips first+last quotes, breaking quoted exe paths.
        $logQuoted = '"' + $LogFile + '"'
        $inner = $CommandLine + ' >> ' + $logQuoted + ' 2>&1'
        $CommandLine = 'cmd /S /C "' + $inner + '"'
    }
    try {
        # Win32_Process.Create defaults to C:\Windows\System32 -- that breaks
        # `python -m <project_module>` since the working directory needs to be
        # the project root for module resolution (and for relative paths like
        # data/, logs/, src/).
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
                              -Arguments @{
                                  CommandLine      = $CommandLine
                                  CurrentDirectory = $root
                              } `
                              -ErrorAction Stop
        if ($r.ReturnValue -eq 0) { return [int]$r.ProcessId }
        Write-Host "  Win32_Process.Create returned $($r.ReturnValue)" -ForegroundColor Red
        return $null
    } catch {
        Write-Host "  Start-Detached failed: $_" -ForegroundColor Red
        return $null
    }
}

# EARLY KILL -- terminate any running bot/dashboard/cluster BEFORE any slow
# pre-step. Pre-fix: startup_recovery ran first and could take 10+ minutes
# (archive gap fill iterates every symbol x timeframe). During those 10 min
# the OLD buggy bot kept running -- on 2026-05-13 that meant 600 seconds of
# spammed sell orders + Gemini-quota burn while the operator waited for
# the restart to advance past [0/6]. Killing first means the bleeding stops
# as soon as the operator types `restart_all.ps1`.
Write-Host ""
Write-Host "[pre-step] Early-kill bot/dashboard/training (stops any runaway loop immediately)..." -ForegroundColor Yellow
$pidFile = Join-Path $root 'data\process_ids.json'
$earlyKilled = @()
if (Test-Path $pidFile) {
    try {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($key in @('bot','dash','monitor','training','realtime','orch','orderbook',
                            'debug','watchdog','sweep_watchdog','cluster_orch')) {
            $pidVal = $pids.$key
            if ($pidVal -and $pidVal -ne 0) {
                try {
                    Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                    $earlyKilled += $pidVal
                } catch {}
            }
        }
    } catch {}
}
# Fallback cmdline scan -- handles bots/dashboards launched outside the PID
# file (e.g. manual Start-Process during a debug session).
Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null | ForEach-Object {
    $cmd = $_.CommandLine
    if ($cmd -match 'src\\main\.py|src/main\.py|-m\s+src\.main|launch_bot|src\\dashboard\\app|-m\s+src\.dashboard|launch_dashboard|launch_training|scripts\.debug_supervisor|scripts\.dashboard_watchdog|scripts\.training_sweep_watchdog|src\.training\.distributed\.orchestrator|src\.monitor\.server|src/monitor/server') {
        if ($_.ProcessId -notin $earlyKilled) {
            try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
            $earlyKilled += $_.ProcessId
        }
    }
}
if ($earlyKilled.Count -gt 0) {
    Write-Host ("  Early-killed {0} process(es) before pre-flight: {1}" -f $earlyKilled.Count, ($earlyKilled -join ', ')) -ForegroundColor Green
} else {
    Write-Host "  No live bot/dashboard found -- clean slate." -ForegroundColor DarkGray
}

# Step 0: ParquetClient store -- file-based, no daemon. Just verifies the
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

# startup_recovery (archive gap fill) is slow -- bounded to 5 minutes so a
# stuck recovery (no-op symbol iteration on a network blip) cannot stall
# the entire restart. Set $env:SKIP_STARTUP_RECOVERY=1 to skip entirely.
if ($env:SKIP_STARTUP_RECOVERY -eq '1') {
    Write-Host "  Skipping startup_recovery (SKIP_STARTUP_RECOVERY=1)." -ForegroundColor DarkGray
} else {
    Write-Host "  Running startup_recovery (archive gap fill, 5-min cap)..."
    $env:PYTHONIOENCODING = 'utf-8'
    $recoveryJob = Start-Job -ScriptBlock {
        param($pyExe, $rootDir)
        & $pyExe -m src.data_ingestion.startup_recovery --archive-only 2>&1 |
            Out-File -Append -FilePath (Join-Path $rootDir 'logs\startup_recovery.log')
    } -ArgumentList $venvPy, $root
    if (Wait-Job $recoveryJob -Timeout 300) {
        Receive-Job $recoveryJob -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  Startup recovery complete." -ForegroundColor Green
    } else {
        Stop-Job $recoveryJob -ErrorAction SilentlyContinue
        Write-Host "  Startup recovery hit 5-min cap -- continuing (the bot's first cycle will pick up any remaining gaps)." -ForegroundColor DarkYellow
    }
    Remove-Job $recoveryJob -Force -ErrorAction SilentlyContinue
}

# Housekeeping: clear DuckDB spill from previous sessions. parquet_store.py
# uses data/cache/duckdb_temp; DuckDB doesn't reliably clean it up on
# Python exit, so it grew to ~14 GB on this machine. Clearing on each
# restart keeps the cache from snowballing again.
$duckTmp = Join-Path $root 'data\cache\duckdb_temp'
if (Test-Path $duckTmp) {
    $duckSize = (Get-ChildItem $duckTmp -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($duckSize -gt 100MB) {
        Write-Host ("  Clearing DuckDB temp spill ({0:N1} GB)..." -f ($duckSize / 1GB)) -ForegroundColor DarkCyan
        Get-ChildItem $duckTmp -Recurse -Force -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

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
        # 2026-05-12 live-validation fix: previously the watchdog PIDs were
        # NOT in the kill list, so each restart_all left the previous
        # watchdog processes alive AND spawned new ones. Live audit caught
        # 2x debug_supervisor + 2x dashboard_watchdog + 2x training_sweep
        # after a single restart. Add them + cluster_orch to the kill list.
        foreach ($key in @('bot','dash','monitor','training','realtime','orch','orderbook',
                            'debug','watchdog','sweep_watchdog','cluster_orch')) {
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
    # 2026-05-12 fix: also kill stray watchdogs + cluster orchestrator on
    # restart so we don't accumulate duplicates across reboots.
    if ($cmd -match 'src\\main\.py|src/main\.py|launch_bot|src\\dashboard\\app|launch_dashboard|launch_training|scripts\.debug_supervisor|scripts\.dashboard_watchdog|scripts\.training_sweep_watchdog|src\.training\.distributed\.orchestrator|src\.monitor\.server|src/monitor/server') {
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

# Helper: launch a .ps1 in the BACKGROUND, fully detached from this shell's
# console group via Start-Detached / WMI. The script's own logging (Python's
# logging module + the launcher's Tee-Object) keeps writing to logs/.
function Start-Window {
    param([string]$Label, [string]$ScriptFile, [string]$LogName = $null)
    Write-Host "  Launching $Label (detached) ..."
    if (-not (Test-Path $ScriptFile)) {
        Write-Host "  WARNING: $ScriptFile not found" -ForegroundColor Red
        return $null
    }
    $cmdLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File "' + $ScriptFile + '"'
    # If LogName given, redirect stdout/stderr at cmd level via Start-Detached.
    # The launcher .ps1 should NOT have its own Out-File pipe -- that races
    # the cmd redirect and locks the file (caused the silent dashboard
    # crashes Apr-May; see launch_dashboard.ps1 / launch_bot.ps1 comments).
    if ($LogName) {
        $logPath = Join-Path $root "logs\$LogName"
        $newPid = Start-Detached -CommandLine $cmdLine -LogFile $logPath
    } else {
        $newPid = Start-Detached -CommandLine $cmdLine
    }
    if (-not $newPid) {
        Write-Host "  $Label failed to start" -ForegroundColor Red
        return $null
    }
    Write-Host "  $Label started (PID $newPid)"
    return [PSCustomObject]@{ Id = $newPid }
}

# Step 0: Monitor server (dashboard)
Write-Host ""
Write-Host "[0/6] Starting Monitor server (http://127.0.0.1:5001)..." -ForegroundColor Yellow
$monRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'src\.monitor\.server|src/monitor/server' }
if ($monRunning) {
    Write-Host "  Monitor already running (PID $($monRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procMonitor = Get-Process -Id $monRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $procMonitor = Start-Window -Label 'Monitor' -ScriptFile (Join-Path $root 'launch_monitor.ps1')
    Start-Sleep -Seconds 3
}
Write-Host "[0/6] Monitor launched." -ForegroundColor Green

# Step 4: ML Training scheduling
# Phase 100d follow-up (2026-05-11) -- DISABLED by default. Pre-fix this
# scheduled launch_training.ps1 to run train_all_models.py directly as a
# subprocess 10 min after restart. train_all_models.py runs LOCALLY (GPU/CPU
# direct), completely bypassing the cluster orchestrator (Phase 100a/b/e).
# Operator saw GPU at 78% while cluster reported "GPU lane idle" because
# this rogue local process was using the GPU outside the cluster's view.
#
# All training paths now go through the cluster orchestrator:
#   - Manual ▶ Train per row -> /api/training/run/<key> (Phase 100a)
#   - Manual ▶ Retrain ALL  -> /api/training/run/all   (Phase 100b)
#   - Auto pipeline         -> pipeline_orchestrator   (Phase 100e)
#
# To re-enable the legacy local-subprocess cron (NOT recommended), set
# $env:AI_TRADER_AUTO_TRAIN = '1' before running restart_all.ps1.
Write-Host ""
$procTraining = $null
if ($env:AI_TRADER_AUTO_TRAIN -eq '1') {
    Write-Host "[4/6] AI_TRADER_AUTO_TRAIN=1 -- scheduling legacy launch_training.ps1 in 10 min" -ForegroundColor Yellow
    $trainingScript = Join-Path $root 'launch_training.ps1'
    $trainingDelay  = 600   # seconds
    if (Test-Path $trainingScript) {
        $cmdLine = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds ' + $trainingDelay + '; & ''' + $trainingScript + '''"'
        $newPid = Start-Detached -CommandLine $cmdLine
        if ($newPid) {
            $procTraining = [PSCustomObject]@{ Id = $newPid }
            Write-Host "  Training will start at $(( Get-Date ).AddSeconds($trainingDelay).ToString('HH:mm:ss')) (PID $newPid, detached)"
        }
    } else {
        Write-Host "  WARNING: launch_training.ps1 not found" -ForegroundColor Red
    }
    Write-Host "[4/6] Legacy training scheduled (override active)." -ForegroundColor Yellow
} else {
    Write-Host "[4/6] Training auto-schedule SKIPPED (cluster handles all training)." -ForegroundColor Green
    Write-Host "      Trigger training via dashboard's Retrain ALL button or per-row Train." -ForegroundColor DarkGray
    Write-Host "      Set AI_TRADER_AUTO_TRAIN=1 to re-enable legacy 10-min local cron." -ForegroundColor DarkGray
}

# Step 4.9: Cluster training orchestrator (:7700) -- REQUIRED by the
# dashboard's /api/cluster/* endpoints, which proxy here. Without this
# the Cluster tab returns 503 "cluster orchestrator unreachable".
# Surfaced 2026-05-12 during live E2E validation.
Write-Host ""
Write-Host "[4.9/6] Starting Cluster Training Orchestrator (:7700)..." -ForegroundColor Yellow
$coRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'src\.training\.distributed\.orchestrator' }
if ($coRunning) {
    Write-Host "  Cluster orchestrator already running (PID $($coRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procClusterOrch = Get-Process -Id $coRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $newPid = Start-Detached -CommandLine "`"$venvPython`" -m src.training.distributed.orchestrator --port 7700" -LogFile "$root\logs\cluster.log"
    if ($newPid) {
        $procClusterOrch = [PSCustomObject]@{ Id = $newPid }
        Write-Host "  Cluster orchestrator started (PID $newPid, detached, :7700)"
    } else {
        $procClusterOrch = $null
    }
}
Write-Host "[4.9/6] Cluster orchestrator ready." -ForegroundColor Green

# Step 5: Dashboard + Bot
Write-Host ""
Write-Host "[5/6] Launching Dashboard and Bot..." -ForegroundColor Yellow
$procDash = Start-Window -Label 'Dashboard' -ScriptFile (Join-Path $root 'launch_dashboard.ps1') -LogName 'dashboard.log'
Start-Sleep -Seconds 2
$procBot  = Start-Window -Label 'Bot'       -ScriptFile (Join-Path $root 'launch_bot.ps1') -LogName 'bot.log'
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

# Step 5.6: FastAPI Control Plane - REMOVED in Phase A11 (2026-05-12).
# All 6 endpoints (/health, /status, /metrics, /control/bot/start,
# /control/bot/stop, /control/training/start) were duplicates of
# dashboard routes / scripts. Removing the service kills one process,
# one auth surface, and one log file. If a downstream tool depends on
# :8100, route it through the dashboard at :5000/api/* instead.

# Step 5.7: Realtime DB Writer (Phase 7) - Binance WS -> QuestDB
Write-Host ""
Write-Host "[5.7/6] Starting Realtime DB Writer (Binance WS -> Parquet)..." -ForegroundColor Yellow
$rtRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'realtime_db_writer' }
if ($rtRunning) {
    Write-Host "  Realtime DB Writer already running (PID $($rtRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procRealtime = Get-Process -Id $rtRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $newPid = Start-Detached -CommandLine "`"$venvPython`" -m src.data_ingestion.realtime_db_writer" -LogFile "$root\logs\realtime_db.log"
    if ($newPid) {
        $procRealtime = [PSCustomObject]@{ Id = $newPid }
        Write-Host "  Realtime DB Writer started (PID $newPid, detached)"
    }
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
    $newPid = Start-Detached -CommandLine "`"$venvPython`" -m src.data_governance.orchestrator" -LogFile "$root\logs\data_orchestrator.log"
    if ($newPid) {
        $procOrch = [PSCustomObject]@{ Id = $newPid }
        Write-Host "  Data Orchestrator started (PID $newPid, detached)"
    }
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
        $newPid = Start-Detached -CommandLine "`"$venvPython`" -m src.data_ingestion.orderbook_collector --symbols `"$obSymbols`" --depth 20 --speed 100ms" -LogFile "$root\logs\orderbook_collector.log"
        if ($newPid) {
            $procOrderbook = [PSCustomObject]@{ Id = $newPid }
            Write-Host "  Orderbook Collector started (PID $newPid, detached, symbols: $obSymbols)"
        }
    }
}
Write-Host "[5.9/6] Orderbook Collector ready." -ForegroundColor Green

# Step 5.91: L2 Orderbook Parquet Writer (X1.2, 2026-05-13)
# Subscribes to the orderflow ZeroMQ channel and persists snapshots to
# data/parquet/_L2/<SYM>/yyyymm=*/  so training pipelines that call
# feature_engineering.add_orderbook_features actually have columns to read.
# Without this, the collector ran but the L2 data was never on disk.
Write-Host ""
Write-Host "[5.91/6] Starting L2 Orderbook Parquet Writer..." -ForegroundColor Yellow
if ($env:OB_WRITER_DISABLED -eq '1') {
    Write-Host "  Skipped (OB_WRITER_DISABLED=1)." -ForegroundColor DarkCyan
} else {
    $obwRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
        Where-Object { $_.CommandLine -match 'orderbook_parquet_writer' }
    if ($obwRunning) {
        Write-Host "  L2 Writer already running (PID $($obwRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    } else {
        $newPid = Start-Detached -CommandLine "`"$venvPython`" -m src.data_ingestion.orderbook_parquet_writer" -LogFile "$root\logs\orderbook_parquet_writer.log"
        if ($newPid) {
            Write-Host "  L2 Writer started (PID $newPid, detached)"
        }
    }
}
Write-Host "[5.91/6] L2 Writer ready." -ForegroundColor Green

# Step 5.95: Debug Supervisor (fine-grained crash detector)
# Polls data/process_ids.json every 5s; on death, captures log tail +
# RSS/CPU snapshot to data/process_deaths.json. Surfaces fresh deaths
# in the banner via _probe_recent_deaths so the user sees crashes
# within seconds. Independent of bot/dashboard so it survives THEIR
# crashes - that's the whole point.
Write-Host ""
Write-Host "[5.95/6] Starting Debug Supervisor (process crash detector)..." -ForegroundColor Yellow
$dsRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'scripts\.debug_supervisor' }
if ($dsRunning) {
    Write-Host "  Debug Supervisor already running (PID $($dsRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procDebug = Get-Process -Id $dsRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $newPid = Start-Detached -CommandLine "`"$venvPython`" -m scripts.debug_supervisor" -LogFile "$root\logs\debug_supervisor.log"
    if ($newPid) {
        $procDebug = [PSCustomObject]@{ Id = $newPid }
        Write-Host "  Debug Supervisor started (PID $newPid, detached)"
    } else {
        $procDebug = $null
    }
}
Write-Host "[5.95/6] Debug Supervisor ready." -ForegroundColor Green

# Step 5.96: Dashboard Watchdog -- keeps :5000 up. Polls /api/state every
# 10s; on FAILURE_THRESHOLD consecutive failures, kills any stale dash
# process and respawns via the same launch_dashboard.ps1 chain. Circuit
# breaker (5 restarts in 10 min) prevents an infinite loop on
# import-time crashes. Independent process so it survives the
# dashboard's death.
Write-Host ""
Write-Host "[5.96/6] Starting Dashboard Watchdog (auto-restart on health-check failure)..." -ForegroundColor Yellow
$wdRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'dashboard_watchdog' }
if ($wdRunning) {
    Write-Host "  Dashboard Watchdog already running (PID $($wdRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procWatchdog = Get-Process -Id $wdRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $newPid = Start-Detached -CommandLine "`"$venvPython`" -m scripts.dashboard_watchdog" -LogFile "$root\logs\dashboard_watchdog.log"
    if ($newPid) {
        $procWatchdog = [PSCustomObject]@{ Id = $newPid }
        Write-Host "  Dashboard Watchdog started (PID $newPid, detached)"
    } else {
        $procWatchdog = $null
    }
}
Write-Host "[5.96/6] Dashboard Watchdog ready." -ForegroundColor Green

# Step 5.97: Training Sweep Watchdog -- keeps the overnight curated sweep
# alive (v3.1). Polls /api/pipeline/status every 60s; respawns the
# orchestrator only when the payload is unchanged for 10+ min AND no
# pipeline_orchestrator process is visible (skip-if-fresh resume picks up
# where the dead attempt died). Never kills in-progress training (per
# operator memory `feedback_dont_relaunch_inflight_training`). Circuit
# breaker: 8 respawns in 6h -> trips, requires manual state clear.
Write-Host ""
Write-Host "[5.97/6] Starting Training Sweep Watchdog (auto-respawn on stall)..." -ForegroundColor Yellow
$swRunning = Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null |
    Where-Object { $_.CommandLine -match 'training_sweep_watchdog' }
if ($swRunning) {
    Write-Host "  Training Sweep Watchdog already running (PID $($swRunning.ProcessId)) - skipping." -ForegroundColor DarkCyan
    $procSweepWatchdog = Get-Process -Id $swRunning.ProcessId -ErrorAction SilentlyContinue
} else {
    $newSwPid = Start-Detached -CommandLine "`"$venvPython`" -m scripts.training_sweep_watchdog" -LogFile "$root\logs\training_sweep_watchdog.log"
    if ($newSwPid) {
        $procSweepWatchdog = [PSCustomObject]@{ Id = $newSwPid }
        Write-Host "  Training Sweep Watchdog started (PID $newSwPid, detached)"
    } else {
        $procSweepWatchdog = $null
    }
}
Write-Host "[5.97/6] Training Sweep Watchdog ready." -ForegroundColor Green

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
$debugId     = if ($procDebug)        { $procDebug.Id        } else { 0 }
$watchdogId  = if ($procWatchdog)     { $procWatchdog.Id     } else { 0 }
$sweepWdId   = if ($procSweepWatchdog) { $procSweepWatchdog.Id } else { 0 }
$clusterOrchId = if ($procClusterOrch) { $procClusterOrch.Id  } else { 0 }
$pidData = @{ bot = $botId; dash = $dashId; monitor = $monId; watchlist = $watchlistId; training = $trainingId; realtime = $realtimeId; orch = $orchId; fastapi = $fastapiId; orderbook = $obId; debug = $debugId; watchdog = $watchdogId; sweep_watchdog = $sweepWdId; cluster_orch = $clusterOrchId; mcp = 0 }
$pidData | ConvertTo-Json | Set-Content (Join-Path $root 'data\process_ids.json')
Write-Host "[6/6] PIDs saved: monitor=$monId  dash=$dashId  bot=$botId  cluster=$clusterOrchId  debug=$debugId  watchdog=$watchdogId  sweep=$sweepWdId  watchlist=$watchlistId  training=$trainingId  realtime=$realtimeId  orch=$orchId  fastapi=$fastapiId  orderbook=$obId" -ForegroundColor Green

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
