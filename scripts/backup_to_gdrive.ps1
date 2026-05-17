# backup_to_gdrive.ps1 -- Sync D:\test 2 to Google Drive via rclone.
#
# Prerequisites (one-time setup):
#   1. Install rclone: https://rclone.org/install/
#      winget install Rclone.Rclone
#   2. Configure Google Drive remote:
#      rclone config
#      -> n (new remote) -> name: gdrive -> type: drive -> follow browser auth
#   3. Test: rclone lsd gdrive:
#
# Usage:
#   PowerShell -ExecutionPolicy Bypass -File "D:\test 2\AI trading assistance\scripts\backup_to_gdrive.ps1"
#
# Scheduled: runs daily at 2:00 AM via install_backup_task.ps1

param(
    [string]$RcloneRemote  = "gdrive",
    [string]$RemoteFolder  = "AI-Trader-Backup/test2",
    [string]$LocalSource   = "D:\test 2",
    [switch]$DryRun        = $false,
    [switch]$Verbose       = $false
)

$ErrorActionPreference = "Stop"
$ScriptStart = Get-Date

$LogDir  = "D:\test 2\AI trading assistance\logs"
$LogFile = Join-Path $LogDir ("backup_gdrive_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }

function Write-Log {
    param([string]$Msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

# Redirect all output to log as well
Start-Transcript -Path $LogFile -Append -NoClobber | Out-Null

Write-Log "============================================================"
Write-Log " Google Drive Backup -- D:\test 2"
Write-Log " Started: $ScriptStart"
Write-Log " Destination: ${RcloneRemote}:${RemoteFolder}"
Write-Log " DryRun: $DryRun"
Write-Log "============================================================"

# ── Check rclone ──────────────────────────────────────────────────────────────
$rclonePath = (Get-Command rclone -ErrorAction SilentlyContinue)?.Source
if (-not $rclonePath) {
    Write-Log "ERROR: rclone not found. Install it:"
    Write-Log "  winget install Rclone.Rclone"
    Write-Log "  Then run: rclone config  (add a remote named 'gdrive')"
    Stop-Transcript | Out-Null
    exit 1
}
Write-Log "rclone: $rclonePath"

# ── Check remote is configured ────────────────────────────────────────────────
$remotes = rclone listremotes 2>&1
if ($remotes -notmatch "${RcloneRemote}:") {
    Write-Log "ERROR: rclone remote '${RcloneRemote}' not configured."
    Write-Log "  Run: rclone config"
    Stop-Transcript | Out-Null
    exit 1
}

# ── Build rclone arguments ────────────────────────────────────────────────────
$Dest = "${RcloneRemote}:${RemoteFolder}"

$rcloneArgs = @(
    "sync",
    $LocalSource,
    $Dest,
    "--log-level", "INFO",
    "--stats", "30s",
    "--transfers", "8",
    "--checkers", "16",
    "--fast-list",
    # Exclude temp/build artifacts -- not worth backing up
    "--exclude", "**/__pycache__/**",
    "--exclude", "**/*.pyc",
    "--exclude", "**/*.pyo",
    "--exclude", "**/venv/**",
    "--exclude", "**/node_modules/**",
    "--exclude", "**/.git/objects/**",      # keep .git metadata, skip large blobs
    "--exclude", "**/.gradle/**",
    "--exclude", "**/build/intermediates/**",
    "--exclude", "**/build/tmp/**",
    "--exclude", "**/.android/**",
    "--exclude", "**/data/cache/**",
    "--exclude", "**/data/raw/**",        # raw csv.gz = duplicate of parquet, 44 GB saved
    "--exclude", "**/*.tmp",
    "--exclude", "**/*.bak",
    "--exclude", "**/*.unicode_fix.bak"
)

if ($DryRun) {
    $rcloneArgs += "--dry-run"
    Write-Log "[DRY RUN] No files will be transferred."
}
if ($Verbose) {
    $rcloneArgs += "--verbose"
}

Write-Log ""
Write-Log "Source : $LocalSource"
Write-Log "Dest   : $Dest"
Write-Log ""

# ── Run sync ──────────────────────────────────────────────────────────────────
Write-Log "[sync] Starting rclone sync..."
$SyncStart = Get-Date

try {
    & rclone @rcloneArgs 2>&1 | ForEach-Object {
        $line = $_
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
        if ($line -match "ERROR|WARNING|Transferred|Checks|Elapsed") {
            Write-Host $line
        }
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "WARNING: rclone exited with code $LASTEXITCODE (partial transfer or remote error)"
    } else {
        Write-Log "[sync] Sync completed successfully."
    }
} catch {
    Write-Log "ERROR during sync: $_"
    Stop-Transcript | Out-Null
    exit 1
}

# ── Update 'last_backup' marker ───────────────────────────────────────────────
if (-not $DryRun) {
    $marker = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    try {
        $marker | rclone rcat "${RcloneRemote}:${RemoteFolder}/.last_backup" 2>&1 | Out-Null
        Write-Log "[marker] Updated .last_backup -> $marker"
    } catch {
        Write-Log "WARNING: Could not write .last_backup marker: $_"
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
$Elapsed = (Get-Date) - $ScriptStart
$ElapsedMin = [math]::Round($Elapsed.TotalMinutes, 1)

Write-Log ""
Write-Log "============================================================"
Write-Log " Backup complete."
Write-Log " Wall time : ${ElapsedMin} min"
Write-Log " Log       : $LogFile"
Write-Log "============================================================"

Stop-Transcript | Out-Null
