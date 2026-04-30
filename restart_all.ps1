$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADER: POWERSHELL AUTO-SETUP"
Write-Host "   Root: $root"
Write-Host "==========================================" -ForegroundColor Cyan

# Step 0: Start QuestDB (primary time-series database — must be up before bot/dashboard)
Write-Host ""
Write-Host "[0/6] Starting QuestDB (primary database)..." -ForegroundColor Yellow
$dockerOk = $true
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "  WARNING: Docker not found — QuestDB will not start. Install Docker Desktop." -ForegroundColor Red
    $dockerOk = $false
}
if ($dockerOk) {
    # Check if container already running
    $qdbRunning = docker ps --filter "name=trading_questdb" --format "{{.Names}}" 2>$null
    if ($qdbRunning -eq "trading_questdb") {
        Write-Host "  QuestDB container already running." -ForegroundColor DarkCyan
    } else {
        # Container exists but stopped → start it; else create it
        $qdbExists = docker ps -a --filter "name=trading_questdb" --format "{{.Names}}" 2>$null
        if ($qdbExists -eq "trading_questdb") {
            Write-Host "  Restarting stopped QuestDB container..."
            docker start trading_questdb 2>&1 | Out-Null
        } else {
            Write-Host "  Creating new QuestDB container..." -ForegroundColor Magenta
            docker run -d `
                --name trading_questdb `
                --restart unless-stopped `
                -p 9000:9000 -p 9009:9009 -p 8812:8812 `
                -v "${root}\data\questdb:/root/.questdb" `
                -e QDB_TELEMETRY_ENABLED=false `
                -e QDB_SHARED_WORKER_COUNT=4 `
                -m 4g `
                questdb/questdb:latest 2>&1 | Out-Null
        }
        # Wait for HTTP health endpoint (up to 30 s)
        Write-Host "  Waiting for QuestDB to be ready..." -NoNewline
        $ready = $false
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep 2
            try {
                $resp = Invoke-WebRequest "http://localhost:9000/health" -TimeoutSec 2 -ErrorAction Stop
                if ($resp.StatusCode -eq 200) { $ready = $true; break }
            } catch {}
            Write-Host "." -NoNewline
        }
        Write-Host ""
        if ($ready) {
            Write-Host "  QuestDB is healthy." -ForegroundColor Green
        } else {
            Write-Host "  WARNING: QuestDB may still be starting — check Docker Desktop." -ForegroundColor Red
        }
    }
    # Ensure schema tables exist (idempotent)
    $venvPy = Join-Path $root 'venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) { $venvPy = "python" }
    Write-Host "  Ensuring DB schema tables exist..."
    & $venvPy -m src.database.schema 2>&1 | Out-Null
    Write-Host "  Schema ready." -ForegroundColor Green
}
Write-Host "[0/6] QuestDB ready." -ForegroundColor Green

# Step 1: Kill ONLY known managed processes (bot, dashboard, monitor, training).
#         Download processes (archive / watchlist) are NOT killed — they finish on their own.
Write-Host ""
Write-Host "[1/6] Terminating managed processes (bot/dashboard/monitor/training only)..." -ForegroundColor Yellow
$pidFile = Join-Path $root 'data\process_ids.json'
$killedPids = @()
if (Test-Path $pidFile) {
    try {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($key in @('bot','dash','monitor','training')) {
            $pid = $pids.$key
            if ($pid -and $pid -ne 0) {
                try {
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    $killedPids += $pid
                    Write-Host "  Stopped $key PID $pid"
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
    if ($cmd -match 'src\\main\.py|src/main\.py|launch_bot|src\\dashboard\\app|launch_dashboard|train_all_models|launch_training') {
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

# Helper: launch a .ps1 in a new window
function Start-Window {
    param([string]$Label, [string]$ScriptFile)
    Write-Host "  Launching $Label ..."
    if (-not (Test-Path $ScriptFile)) {
        Write-Host "  WARNING: $ScriptFile not found" -ForegroundColor Red
        return $null
    }
    $argStr = '-NoExit -ExecutionPolicy Bypass -File "' + $ScriptFile + '"'
    $proc = Start-Process powershell -ArgumentList $argStr -PassThru
    Write-Host "  $Label started (PID $($proc.Id))"
    return $proc
}

# Step 0: Monitor server (dashboard)
Write-Host ""
Write-Host "[0/6] Starting Monitor server (http://127.0.0.1:5001)..." -ForegroundColor Yellow
$procMonitor = Start-Window -Label 'Monitor' -ScriptFile (Join-Path $root 'launch_monitor.ps1')
Start-Sleep -Seconds 3
Write-Host "[0/6] Monitor launched." -ForegroundColor Green

# Step 4: ML Training — deferred 10 minutes after restart so the bot stabilises first
Write-Host ""
Write-Host "[4/6] Scheduling ML Training to start in 10 minutes..." -ForegroundColor Yellow
$trainingScript = Join-Path $root 'launch_training.ps1'
$trainingDelay  = 600   # seconds
$procTraining = $null
if (Test-Path $trainingScript) {
    $argStr = "-NoExit -ExecutionPolicy Bypass -Command `"Start-Sleep -Seconds $trainingDelay; & '$trainingScript'`""
    $procTraining = Start-Process powershell -ArgumentList $argStr -PassThru
    Write-Host "  Training will start at $(( Get-Date ).AddSeconds($trainingDelay).ToString('HH:mm:ss')) (PID $($procTraining.Id))"
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

# Step 6: Save PIDs
Write-Host ""
Write-Host "[6/6] Saving process IDs..." -ForegroundColor Yellow
$monId       = if ($procMonitor)   { $procMonitor.Id   } else { 0 }
$dashId      = if ($procDash)      { $procDash.Id      } else { 0 }
$botId       = if ($procBot)       { $procBot.Id       } else { 0 }
$watchlistId = if ($procWatchlist) { $procWatchlist.Id } else { 0 }
$trainingId  = if ($procTraining)  { $procTraining.Id  } else { 0 }
$pidData = @{ bot = $botId; dash = $dashId; monitor = $monId; watchlist = $watchlistId; training = $trainingId; mcp = 0 }
$pidData | ConvertTo-Json | Set-Content (Join-Path $root 'data\process_ids.json')
Write-Host "[6/6] PIDs saved: monitor=$monId  dash=$dashId  bot=$botId  watchlist=$watchlistId  training=$trainingId" -ForegroundColor Green

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
