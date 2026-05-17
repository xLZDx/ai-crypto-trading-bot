# install_backup_task.ps1 -- Register daily 2AM Google Drive backup in Windows Task Scheduler.
#
# Run ONCE as Administrator:
#   Right-click PowerShell -> Run as Administrator
#   cd "D:\test 2\AI trading assistance"
#   PowerShell -ExecutionPolicy Bypass -File scripts\install_backup_task.ps1
#
# To remove the task:
#   PowerShell -ExecutionPolicy Bypass -File scripts\install_backup_task.ps1 -Uninstall
#
# To run the backup manually now:
#   Start-ScheduledTask -TaskName "AI-Trader-GDrive-Backup"

param(
    [switch]$Uninstall = $false
)

$TaskName    = "AI-Trader-GDrive-Backup"
$ScriptPath  = "D:\test 2\AI trading assistance\scripts\backup_to_gdrive.ps1"
$TriggerTime = "02:00"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' removed."
    exit 0
}

# ── Verify script exists ──────────────────────────────────────────────────────
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Backup script not found: $ScriptPath"
    exit 1
}

# ── Build task definition ─────────────────────────────────────────────────────
$Action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$ScriptPath`""

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $TriggerTime

# Run whether user is logged on or not, with highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -WakeToRun $false `
    -MultipleInstances IgnoreNew

# ── Register (replace if exists) ─────────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "Daily 2AM backup of D:\test 2 to Google Drive via rclone" | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host " Task registered: $TaskName"
Write-Host " Runs daily at  : $TriggerTime"
Write-Host " Script         : $ScriptPath"
Write-Host "============================================================"
Write-Host ""
Write-Host "Commands:"
Write-Host "  Run now   : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Check log : Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host "  Remove    : PowerShell -File scripts\install_backup_task.ps1 -Uninstall"
Write-Host ""

# ── Verify registration ───────────────────────────────────────────────────────
$task = Get-ScheduledTask -TaskName $TaskName
Write-Host "Status: $($task.State)"
