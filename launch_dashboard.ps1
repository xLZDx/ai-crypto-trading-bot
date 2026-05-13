$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trading Dashboard'
Set-Location $root

# Phase 11 - UTF-8 logs (avoid PowerShell's default UTF-16 from Tee-Object,
# which broke the live-log viewer). Plus optional dedicated IP binding.
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:DASHBOARD_BIND_HOST) { $env:DASHBOARD_BIND_HOST = '0.0.0.0' }
if (-not $env:DASHBOARD_BIND_PORT) { $env:DASHBOARD_BIND_PORT = '5000' }
Write-Host "[dashboard] BIND $($env:DASHBOARD_BIND_HOST):$($env:DASHBOARD_BIND_PORT)"

# Logging: just exec python and let stdout/stderr inherit. When this script
# is launched via restart_all.ps1's Start-Detached helper, the cmd-level
# `>> dashboard.log 2>&1` redirect captures everything. The previous
# Out-File pipeline created a SECOND writer to dashboard.log, racing with
# the cmd redirect -- the resulting "file in use" errors crashed the
# dashboard within seconds of every spawn.
& $python (Join-Path $root 'src\dashboard\app.py')
