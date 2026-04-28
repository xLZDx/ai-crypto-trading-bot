$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADER: POWERSHELL AUTO-SETUP"
Write-Host "   Root: $root"
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Kill old processes
Write-Host ""
Write-Host "[1/6] Terminating old background processes..." -ForegroundColor Yellow
$pidFile = Join-Path $root 'data\process_ids.json'
if (Test-Path $pidFile) {
    Write-Host "  Found PID file, removing it."
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "  No PID file found."
}
Write-Host "  Killing python.exe (10s timeout)..."
$killJob = Start-Job -ScriptBlock { taskkill /F /IM python.exe 2>&1 | Out-Null }
$done = Wait-Job $killJob -Timeout 10
if ($done) {
    Write-Host "  Kill completed."
} else {
    Write-Host "  Kill timed out (zombie processes ignored)."
}
Remove-Job $killJob -Force -ErrorAction SilentlyContinue
Write-Host "[1/6] Processes terminated." -ForegroundColor Green

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
Write-Host "  Directories ready."

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

# Step 0: Monitor server
Write-Host ""
Write-Host "[0/6] Starting Monitor server (http://127.0.0.1:5001)..." -ForegroundColor Yellow
$procMonitor = Start-Window -Label 'Monitor' -ScriptFile (Join-Path $root 'launch_monitor.ps1')
Start-Sleep -Seconds 3
Write-Host "[0/6] Monitor launched." -ForegroundColor Green

# Step 4: ML Training
Write-Host ""
Write-Host "[4/6] Launching ML Training..." -ForegroundColor Yellow
Start-Window -Label 'Training' -ScriptFile (Join-Path $root 'launch_training.ps1') | Out-Null
Start-Sleep -Seconds 2
Write-Host "[4/6] Training launched." -ForegroundColor Green

# Step 5: Dashboard + Bot
Write-Host ""
Write-Host "[5/6] Launching Dashboard and Bot..." -ForegroundColor Yellow
$procDash = Start-Window -Label 'Dashboard' -ScriptFile (Join-Path $root 'launch_dashboard.ps1')
Start-Sleep -Seconds 2
$procBot  = Start-Window -Label 'Bot'       -ScriptFile (Join-Path $root 'launch_bot.ps1')
Start-Sleep -Seconds 2
Write-Host "[5/6] Dashboard and Bot launched." -ForegroundColor Green

# Step 6: Save PIDs
Write-Host ""
Write-Host "[6/6] Saving process IDs..." -ForegroundColor Yellow
$monId  = if ($procMonitor) { $procMonitor.Id } else { 0 }
$dashId = if ($procDash)    { $procDash.Id    } else { 0 }
$botId  = if ($procBot)     { $procBot.Id     } else { 0 }
$pidData = @{ bot = $botId; dash = $dashId; monitor = $monId; mcp = 0 }
$pidData | ConvertTo-Json | Set-Content (Join-Path $root 'data\process_ids.json')
Write-Host "[6/6] PIDs saved: monitor=$monId  dash=$dashId  bot=$botId" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   ALL PROCESSES STARTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "   Monitor   -> http://127.0.0.1:5001"   -ForegroundColor Cyan
Write-Host "   Dashboard -> http://127.0.0.1:5000"   -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "This window stays open for your reference." -ForegroundColor White
Read-Host "Press Enter to close"
