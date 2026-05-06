# Zombie watchdog for AI Trading Assistance.
#
# Run via the AITradingZombieWatchdog scheduled task every 10 minutes.
# Strictly project-scoped: only touches python.exe processes whose command
# line references this project root. Never touches Claude Code, VSCode,
# Chrome, the Android emulator, or any other process on the system.
#
# Behavior:
#   1. Group project python processes by exact command line. For each group
#      with duplicates, keep the newest (matches restart_all.ps1 pattern)
#      and kill the older ones.
#   2. Kill orphaned joblib loky resource_tracker workers whose parent PID
#      is no longer alive.
#   3. Skip any process younger than $graceSeconds to avoid races with
#      restart_all.ps1 and stop_all.ps1.
#
# Always exits 0 so Task Scheduler doesn't flag missed-run errors.

$ErrorActionPreference = "Continue"

$projectRoot   = "D:\test 2\AI trading assistance"
$logDir        = Join-Path $projectRoot "logs"
$logFile       = Join-Path $logDir "zombie_watchdog.log"
$graceSeconds  = 60

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $Message" | Out-File -Append -FilePath $logFile -Encoding utf8
}

function Truncate {
    param([string]$Text, [int]$Max = 120)
    if (-not $Text) { return "" }
    if ($Text.Length -le $Max) { return $Text }
    return $Text.Substring(0, $Max)
}

$now = Get-Date

# Project scope filter — single source of truth. Matches venv launches and
# `-m src.foo` invocations because both reference the project path on the
# command line. Excludes everything else by construction.
$projectProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -like "*$projectRoot*") }

if (-not $projectProcs) {
    # Don't spam the log — heartbeat once an hour is enough when clean.
    if ((Get-Date).Minute -lt 10) {
        Write-Log "scan: 0 project python processes"
    }
    exit 0
}

$killed = 0

# 1) Command-line dedup. Keep newest, kill older.
#
# CRITICAL: skip parent↔child pairs. Windows venv python.exe is a thin
# launcher that exec's the real interpreter — both processes have IDENTICAL
# CommandLine in WMI but ONE IS THE PARENT OF THE OTHER. Killing the parent
# breaks the child's stdin/stdout pipe → BrokenPipeError → silent death.
# That bug masqueraded as "dashboard keeps crashing silently" all session.
# Only flag as duplicates when neither member is an ancestor of the other.
$groups = $projectProcs | Group-Object CommandLine
foreach ($g in $groups) {
    if ($g.Count -le 1) { continue }
    $sorted     = $g.Group | Sort-Object CreationDate -Descending

    # Check parent/child relationships. If any process in the group is a
    # parent of another in the same group, this is a venv launcher pair —
    # NOT a duplicate restart. Skip the whole group.
    $pids = $sorted | ForEach-Object { $_.ProcessId }
    $isLauncherPair = $false
    foreach ($p in $sorted) {
        if ($pids -contains $p.ParentProcessId) {
            $isLauncherPair = $true
            break
        }
    }
    if ($isLauncherPair) {
        Write-Log "skip-launcher-pair count=$($g.Count) cmd=$(Truncate $g.Name)"
        continue
    }

    $keeper     = $sorted[0]
    $duplicates = $sorted | Select-Object -Skip 1
    foreach ($p in $duplicates) {
        $age = ($now - $p.CreationDate).TotalSeconds
        if ($age -lt $graceSeconds) {
            Write-Log "skip-young PID=$($p.ProcessId) age=$([math]::Round($age,1))s cmd=$(Truncate $p.CommandLine)"
            continue
        }
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Log "killed-dup PID=$($p.ProcessId) keeping=$($keeper.ProcessId) cmd=$(Truncate $p.CommandLine)"
            $killed++
        } catch {
            Write-Log "fail-dup PID=$($p.ProcessId) err=$($_.Exception.Message)"
        }
    }
}

# 2) Orphaned joblib resource_tracker workers — parent dead.
#    Command pattern: ... main(<PARENT_PID>, <verbose>)
$jobLibs = $projectProcs | Where-Object { $_.CommandLine -like "*resource_tracker*" }
foreach ($j in $jobLibs) {
    if ($j.CommandLine -match 'main\((\d+),') {
        $parentPid = [int]$Matches[1]
        $parentAlive = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
        if ($parentAlive) { continue }
        $age = ($now - $j.CreationDate).TotalSeconds
        if ($age -lt $graceSeconds) {
            Write-Log "skip-young-orphan PID=$($j.ProcessId) deadParent=$parentPid age=$([math]::Round($age,1))s"
            continue
        }
        try {
            Stop-Process -Id $j.ProcessId -Force -ErrorAction Stop
            Write-Log "killed-orphan PID=$($j.ProcessId) deadParent=$parentPid"
            $killed++
        } catch {
            Write-Log "fail-orphan PID=$($j.ProcessId) err=$($_.Exception.Message)"
        }
    }
}

if ($killed -gt 0) {
    Write-Log "scan: killed=$killed survivors=$(($projectProcs.Count) - $killed)"
} else {
    if ((Get-Date).Minute -lt 10) {
        Write-Log "scan: clean projectPids=$($projectProcs.Count)"
    }
}

exit 0
