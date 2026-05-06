# ──────────────────────────────────────────────────────────────────────────────
# launch_fastapi.ps1 - Start the institutional control plane on :8100.
#
# Uses the same plain-shell pattern as launch_bot.ps1 / launch_dashboard.ps1
# (Start-Process / pipeline) so heuristic AV scanners (Norton IDP.Generic
# and similar) don't flag this as a process-injector pattern.
# Read-mostly FastAPI surface for /health, /status, /metrics, and a small
# /control/* set for bot/training lifecycle.
# ──────────────────────────────────────────────────────────────────────────────

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trader FastAPI Control Plane'
Set-Location $root

$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:FASTAPI_BIND_HOST) { $env:FASTAPI_BIND_HOST = '127.0.0.1' }
if (-not $env:FASTAPI_BIND_PORT) { $env:FASTAPI_BIND_PORT = '8100' }
Write-Host "[fastapi] BIND $($env:FASTAPI_BIND_HOST):$($env:FASTAPI_BIND_PORT)"

# Idempotent - if /health responds, don't double-launch.
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$($env:FASTAPI_BIND_PORT)/health" `
                           -UseBasicParsing -TimeoutSec 1.5 -ErrorAction Stop
    if ($r.StatusCode -eq 200) {
        Write-Host "FastAPI already running on :$($env:FASTAPI_BIND_PORT) - skipping launch."
        exit 0
    }
} catch { }

# Logging: when launched via restart_all.ps1's Start-Detached, the cmd-level
# `>> fastapi.log 2>&1` redirect captures everything. The previous Out-File
# pipeline raced the cmd redirect → IOException → silent crash (same bug
# as launch_bot.ps1 / launch_dashboard.ps1; see commit 76d521b).
& $python -m src.server.control_plane
