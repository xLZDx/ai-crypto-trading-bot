# Remove the AITradingZombieWatchdog scheduled task.

$ErrorActionPreference = "Stop"
$taskName = "AITradingZombieWatchdog"

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task '$taskName'." -ForegroundColor Green
} else {
    Write-Host "No task named '$taskName' is registered." -ForegroundColor DarkGray
}
