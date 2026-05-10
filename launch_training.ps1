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

# Pass any arguments from this script directly to the python script.
# The script-block wrap is intentional - keeps stderr from being
# surfaced as NativeCommandError noise to the operator console.
& {
    & $python -u (Join-Path $root 'src\engine\train_all_models.py') $args 2>&1
} | Out-String -Stream | Tee-Object -FilePath (Join-Path $root 'logs\training.log') -Append

# Forward the python exit code so callers (CI, schedulers) see the real
# success/failure status rather than just "PowerShell returned 0."
exit $LASTEXITCODE
