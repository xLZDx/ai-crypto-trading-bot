$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$host.UI.RawUI.WindowTitle = 'AI Trading Bot'
Set-Location $root

. (Join-Path $root 'setup_env.ps1')

# Phase 11 - UTF-8 logs + optional dedicated IP binding for any local API.
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:BOT_BIND_HOST) { $env:BOT_BIND_HOST = '0.0.0.0' }
Write-Host "[bot] BIND $($env:BOT_BIND_HOST)"

# Logging: when launched via restart_all.ps1's Start-Detached, the cmd-level
# `>> bot.log 2>&1` redirect captures everything. The previous Out-File
# pipeline raced the cmd redirect -> IOException ("file used by another
# process") -> silent crash within seconds. Just exec python directly.
& $python (Join-Path $root 'src\main.py')
