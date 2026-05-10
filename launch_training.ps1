# 2026-05-10 fix - PowerShell 5.1+ wraps every stderr write from a native
# command as a NativeCommandError, even harmless Python INFO log lines
# routed through stderr (logging default). Three layers of defense:
#   1. ErrorActionPreference=Continue at script scope so a non-terminating
#      error never escalates to terminating
#   2. PS7+: PSNativeCommandUseErrorActionPreference=false so native cmd
#      stderr is not auto-thrown (no-op on 5.1)
#   3. Script-block wrap around the python call so the 2>&1 redirect
#      catches stderr BEFORE PowerShell inspects it as an error stream.
$ErrorActionPreference = 'Continue'
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'ML Training'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

# 2026-05-10 fix - also check the lock here so a manual relaunch from the
# operator's shell doesn't bypass the gate. The Python side has its own
# acquisition logic; this is a fast-fail in front of it. Pass --force to
# override (passed straight through to the python script).
$lockPath = Join-Path $root 'data\train_all_models.lock'
$force    = ($args -contains '--force')
if ((Test-Path $lockPath) -and (-not $force)) {
    try {
        $lockJson = Get-Content $lockPath -Raw | ConvertFrom-Json
        $prevPid  = [int]$lockJson.pid
        $alive    = $false
        if ($prevPid -gt 0) {
            $proc = Get-Process -Id $prevPid -ErrorAction SilentlyContinue
            if ($proc) { $alive = $true }
        }
        if ($alive) {
            Write-Host "" -ForegroundColor Yellow
            Write-Host "Another train_all_models.py is already running (pid=$prevPid)." -ForegroundColor Yellow
            Write-Host "Started: $($lockJson.started_iso)" -ForegroundColor Yellow
            Write-Host "Pass --force to spawn a parallel run anyway, or kill that pid first." -ForegroundColor Yellow
            Write-Host ""
            exit 2
        }
    } catch {
        Write-Host "Could not parse lock file ($($_.Exception.Message)) - proceeding." -ForegroundColor DarkYellow
    }
}

# Pass any arguments from this script directly to the python script.
# The script-block wrap is intentional - keeps stderr from being
# surfaced as NativeCommandError noise to the operator console.
& {
    & $python -u (Join-Path $root 'src\engine\train_all_models.py') $args 2>&1
} | Out-String -Stream | Tee-Object -FilePath (Join-Path $root 'logs\training.log') -Append

# Forward the python exit code so callers (CI, schedulers) see the real
# success/failure status rather than just "PowerShell returned 0."
exit $LASTEXITCODE
