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

$logFile = Join-Path $logDir 'bot.log'
# Each line is treated as plain text (write-output, not write-host) so PowerShell
# doesn't paint Python's stderr with the red NativeCommandError style.
& $python (Join-Path $root 'src\main.py') 2>&1 |
    ForEach-Object {
        $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { [string]$_ }
        Write-Output $line
        $line | Out-File -FilePath $logFile -Append -Encoding utf8
    }
