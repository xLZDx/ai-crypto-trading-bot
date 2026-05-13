$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADING ASSISTANCE - STOP ALL" -ForegroundColor Cyan
Write-Host "   Root: $root"
Write-Host "==========================================" -ForegroundColor Cyan

$pidFile = Join-Path $root 'data\process_ids.json'
$killed = @()

if (Test-Path $pidFile) {
    Write-Host ""
    Write-Host "[1/3] Reading $pidFile ..." -ForegroundColor Yellow
    try {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($key in @('bot','dash','monitor','training','watchlist','realtime','orch','fastapi','orderbook','mcp')) {
            $pidVal = $pids.$key
            if ($pidVal -and $pidVal -ne 0) {
                try {
                    $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
                    if ($proc) {
                        Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                        Write-Host "  Stopped $key (PID $pidVal)" -ForegroundColor Green
                        $killed += $pidVal
                    } else {
                        Write-Host "  $key (PID $pidVal) - already gone" -ForegroundColor DarkGray
                    }
                } catch {
                    Write-Host "  $key (PID $pidVal) - error: $_" -ForegroundColor Red
                }
            }
        }
    } catch {
        Write-Host "  Could not parse PID file: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[1/3] No data/process_ids.json - skipping PID-based shutdown." -ForegroundColor DarkYellow
}

# Fallback - find any python.exe whose command line contains our scripts.
Write-Host ""
Write-Host "[2/3] Sweeping stray python.exe processes ..." -ForegroundColor Yellow
$pattern = 'src\\main\.py|src/main\.py|launch_bot|src\\dashboard\\app|launch_dashboard|train_all_models|launch_training|realtime_db_writer|data_governance.orchestrator|watchlist_downloader|telegram_persistor|orderbook_collector|src\.server\.control_plane'
$count = 0
Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null | ForEach-Object {
    $cmd = $_.CommandLine
    if ($cmd -and ($cmd -match $pattern) -and ($_.ProcessId -notin $killed)) {
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped stray PID $($_.ProcessId): $($cmd.Substring(0,[Math]::Min(80,$cmd.Length)))" -ForegroundColor Green
            $count++
        } catch {}
    }
}
if ($count -eq 0) { Write-Host "  No strays found." -ForegroundColor DarkGray }

# GPU sanity check - confirm no python is using the NVIDIA GPU.
# Without this, users see Task Manager showing "GPU 1: 94%" and assume training
# is still running, when often it is just a game / DWM / browser using the GPU.
Write-Host ""
Write-Host "[3/4] GPU sanity check (any python/training process still on the GPU?)..." -ForegroundColor Yellow
$nvSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvSmi) {
    $gpuProcs = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null
    $ourGpuHits = @()
    foreach ($line in ($gpuProcs -split "`n")) {
        if ($line -match '(?i)(python|train_|launch_|src\.)') { $ourGpuHits += $line.Trim() }
    }
    if ($ourGpuHits.Count -eq 0) {
        Write-Host "  OK - no python/training processes on the GPU." -ForegroundColor Green
        Write-Host "  Any GPU usage you see now is from games / browser / DWM / Razer / Citrix." -ForegroundColor DarkGray
    } else {
        Write-Host "  WARNING - these GPU processes appear to be ours:" -ForegroundColor Red
        $ourGpuHits | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    }
} else {
    Write-Host "  nvidia-smi not on PATH - skipping GPU check." -ForegroundColor DarkGray
}

# Parquet store is file-based - nothing to stop. Was a QuestDB note pre-migration.
Write-Host ""
Write-Host "[4/4] Parquet store is file-based - no daemon to stop." -ForegroundColor DarkGray

# Clean PID file so the next restart starts clean.
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "Removed $pidFile" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   ALL PROCESSES STOPPED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
