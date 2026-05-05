# ------------------------------------------------------------------------------
# clean_restart.ps1 - hard restart, single visible terminal window.
#
# What it does:
#   1. Kills every Python + PowerShell process whose path/title belongs to
#      this project (so leftovers from previous restart_all kicks die too).
#   2. Closes any visible PowerShell / conhost / Windows Terminal windows
#      whose title matches "AI Trading" (the launchers set those titles).
#   3. Opens ONE new visible PowerShell window running restart_all.ps1 with
#      -NoExit so progress stays on screen and the window doesn't auto-close.
#
# After it returns the only AI-Trader window on screen is the one it just
# spawned. Read the live progress there.
# ------------------------------------------------------------------------------

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$myPid = $PID

Write-Host "[clean_restart] killing project processes..." -ForegroundColor Yellow

# Step 1: Python processes whose executable is in our venv, or cmdline mentions
# this project root. WMI gives us the cmdline, which Get-Process doesn't.
$cwdPattern = "*AI trading assistance*"
$pyProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^(python|py|pythonw)\.exe$' -and
        ($_.ExecutablePath -like $cwdPattern -or $_.CommandLine -like $cwdPattern)
    }
foreach ($p in $pyProcs) {
    Write-Host ("  python pid={0}  {1}" -f $p.ProcessId, ($p.CommandLine -replace '^.{0,80}', '')[0..120] -join '') -ForegroundColor DarkGray
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

# Step 2: PowerShell windows with AI Trading titles (launch_bot.ps1 +
# launch_dashboard.ps1 set $host.UI.RawUI.WindowTitle on boot).
$psProcs = Get-Process powershell, pwsh -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $myPid -and $_.MainWindowTitle -match 'AI Trad' }
foreach ($p in $psProcs) {
    Write-Host ("  powershell pid={0}  '{1}'" -f $p.Id, $p.MainWindowTitle) -ForegroundColor DarkGray
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}

# Step 3: detached background PowerShell processes spawned by restart_all
# (-WindowStyle Hidden, so no MainWindowTitle). Match by command line.
$psBgProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^powershell\.exe$' -and
        $_.ProcessId -ne $myPid -and
        $_.CommandLine -like $cwdPattern
    }
foreach ($p in $psBgProcs) {
    Write-Host ("  ps-bg pid={0}" -f $p.ProcessId) -ForegroundColor DarkGray
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

# Wipe stale PID + lock files so the new run starts clean.
Remove-Item (Join-Path $root 'data\process_ids.json') -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $root 'data\dash_pid.txt')     -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "[clean_restart] launching fresh restart_all.ps1 in a new window..." -ForegroundColor Cyan

# Step 4: open ONE new visible PowerShell with restart_all.ps1.
# -NoExit keeps the window open so the user sees progress + can read errors.
# Set the window title so the next clean_restart can recognise + close it.
$cmd = "`$host.UI.RawUI.WindowTitle = 'AI Trading - restart_all'; & '$root\restart_all.ps1'"
Start-Process powershell -ArgumentList '-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd

Write-Host "[clean_restart] done. Switch to the new 'AI Trading - restart_all' window." -ForegroundColor Green
