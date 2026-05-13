$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "   TFT MODEL: Force Kill & Relaunch with Custom Args"
Write-Host "======================================================" -ForegroundColor Cyan

# --- Step 1: Find and kill any existing training process ---
Write-Host ""
Write-Host "[1/2] Terminating any active ML training process..." -ForegroundColor Yellow
$pidFile = Join-Path $root 'data\process_ids.json'
$killed = $false
if (Test-Path $pidFile) {
    try {
        $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
        $pidVal = $pids.training
        if ($pidVal -and $pidVal -ne 0) {
            Write-Host "  Stopping 'training' process from process_ids.json (PID $pidVal)..."
            Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
            $killed = $true
        }
    } catch {
        Write-Host "  Could not parse PID file: $_" -ForegroundColor DarkYellow
    }
}

# Fallback: find any python process running a known training script
Get-WmiObject Win32_Process -Filter "Name='python.exe'" 2>$null | ForEach-Object {
    $cmd = $_.CommandLine
    if ($cmd -match 'train_all_models|launch_training') {
        Write-Host "  Stopping stray training process (PID $($_.ProcessId))..."
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
        $killed = $true
    }
}
if ($killed) { Write-Host "[1/2] Training process terminated." -ForegroundColor Green }
else { Write-Host "[1/2] No active training process found to terminate." -ForegroundColor DarkCyan }

Start-Sleep -Seconds 2

# --- Step 2: Relaunch training with specific arguments ---
Write-Host ""
Write-Host "[2/2] Relaunching TFT training with --epochs 3..." -ForegroundColor Yellow
Write-Host "The training output will now be displayed directly in this terminal." -ForegroundColor Cyan
Write-Host "The script will block here until training is complete. Press Ctrl+C to stop." -ForegroundColor Cyan
$trainingScript = Join-Path $root 'launch_training.ps1'

# Run the training script directly in this window to show live progress.
& $trainingScript --model tft --epochs 3