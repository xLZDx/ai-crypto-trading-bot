# Register the AITradingZombieWatchdog scheduled task (every 10 minutes).
# Idempotent — safe to re-run; recreates the task each time.
#
# Requires: PowerShell 5+ with the ScheduledTasks module (built-in on
# Windows 10/11). Standard user rights are sufficient for a per-user task.

$ErrorActionPreference = "Stop"

$taskName     = "AITradingZombieWatchdog"
$projectRoot  = "D:\test 2\AI trading assistance"
$watchdogPath = Join-Path $projectRoot "scripts\zombie_watchdog.ps1"

if (-not (Test-Path $watchdogPath)) {
    Write-Error "Watchdog script not found at $watchdogPath"
    exit 1
}

$pwsh = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Path
if (-not $pwsh) {
    $pwsh = (Get-Command powershell.exe).Path
    Write-Warning "pwsh.exe not on PATH; using Windows PowerShell at $pwsh"
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$taskName'..." -ForegroundColor DarkYellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute $pwsh `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$watchdogPath`""

# Trigger: start in 1 minute, repeat every 10 minutes for ~10 years.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew

# LogonType Interactive — runs while the user is logged in (which is when
# the bot/dashboard run anyway). Avoids storing credentials.
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Kills duplicate project python processes every 10 minutes." | Out-Null

Write-Host "Scheduled task '$taskName' registered." -ForegroundColor Green
Write-Host "First run    : $((Get-Date).AddMinutes(1).ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Interval     : every 10 minutes"
Write-Host "Watchdog log : $(Join-Path $projectRoot 'logs\zombie_watchdog.log')"
Write-Host ""
Write-Host "Manage with:"
Write-Host "  Get-ScheduledTask -TaskName $taskName"
Write-Host "  Start-ScheduledTask -TaskName $taskName    # run now"
Write-Host "  scripts\uninstall_zombie_watchdog.ps1      # remove"
