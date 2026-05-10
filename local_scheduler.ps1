# ──────────────────────────────────────────────────────────────────────────────
# local_scheduler.ps1 - register/list/unregister LOCAL Windows scheduled tasks.
#
# All execution stays on this machine. No cloud, no remote agents, no API
# calls outside loopback. Wraps `schtasks.exe` so the task survives reboots
# and runs under the current user.
#
# Usage:
#   .\local_scheduler.ps1 register   -Name "AI-Trader-TFTCheck" -At "21:30"
#   .\local_scheduler.ps1 register   -Name "AI-Trader-TFTCheck" -EveryMinutes 30
#   .\local_scheduler.ps1 register   -Name "AI-Trader-TFTCheck" -Once "2026-05-02T21:30:00"
#   .\local_scheduler.ps1 list
#   .\local_scheduler.ps1 unregister -Name "AI-Trader-TFTCheck"
#   .\local_scheduler.ps1 run        -Name "AI-Trader-TFTCheck"
# ──────────────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('register','list','unregister','run')]
    [string]$Action,

    [string]$Name = 'AI-Trader-TFTCheck',
    [string]$Script,            # absolute or repo-relative .py path
    [string]$At,                # "HH:MM" - daily
    [int]$EveryMinutes,         # repeat interval in minutes
    [string]$Once               # one-time run, "YYYY-MM-DDTHH:MM:SS" local time
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python      = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
$DefaultPy   = Join-Path $ProjectRoot 'scripts\check_training_status.py'

function Resolve-ScriptPath {
    param([string]$P)
    if (-not $P) { return $DefaultPy }
    if ([System.IO.Path]::IsPathRooted($P)) { return $P }
    return (Join-Path $ProjectRoot $P)
}

function Action-Register {
    $scriptPath = Resolve-ScriptPath $Script
    if (-not (Test-Path $scriptPath)) {
        Write-Host "ERROR: script not found: $scriptPath" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $Python)) {
        Write-Host "ERROR: venv python missing: $Python" -ForegroundColor Red
        exit 1
    }

    # schtasks.exe /TR has a famously broken parser: it re-tokenizes its
    # argument value AFTER PowerShell has already stripped outer quotes, so
    # any embedded space in the project path ("D:\test 2\AI trading…") is
    # mis-split (yields "Invalid argument/option - '2\AI'"). Multiple cmd /c
    # quoting variants don't survive this round-trip reliably.
    #
    # Bullet-proof workaround: write a tiny wrapper .cmd file that contains
    # the actual command, and point /TR at the wrapper — schtasks then only
    # has to handle ONE token. Using .NET WriteAllText (not Set-Content) so
    # PowerShell doesn't mangle embedded quotes / line endings.
    $wrapperDir = Join-Path $ProjectRoot 'scripts\scheduled'
    if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Force -Path $wrapperDir | Out-Null }
    $safeName = ($Name -replace '[^A-Za-z0-9_.-]', '_')
    $wrapperPath = Join-Path $wrapperDir ("$safeName.cmd")
    $crlf = [char]13 + [char]10
    $wrapper = '@echo off' + $crlf `
             + 'cd /d "' + $ProjectRoot + '"' + $crlf `
             + '"' + $Python + '" "' + $scriptPath + '" --quiet' + $crlf
    [System.IO.File]::WriteAllText($wrapperPath, $wrapper, [System.Text.Encoding]::ASCII)
    $tr = '"' + $wrapperPath + '"'

    # Pick schedule mode based on which parameter was supplied.
    $schedArgs = @()
    if ($Once) {
        # ONE-TIME: schtasks /SC ONCE /SD <date> /ST <time>
        try {
            $dt = [DateTime]::Parse($Once)
        } catch {
            Write-Host "ERROR: -Once must parse as DateTime, got '$Once'" -ForegroundColor Red
            exit 1
        }
        $sd = $dt.ToString('MM/dd/yyyy')
        $st = $dt.ToString('HH:mm')
        $schedArgs = @('/SC','ONCE','/SD', $sd, '/ST', $st)
        Write-Host "[register] one-time: $sd $st (local)"
    }
    elseif ($EveryMinutes -gt 0) {
        # RECURRING every N minutes (Windows Task Scheduler supports this via
        # /SC MINUTE /MO N - runs forever once started).
        $schedArgs = @('/SC','MINUTE','/MO', "$EveryMinutes")
        Write-Host "[register] recurring every $EveryMinutes minute(s)"
    }
    elseif ($At) {
        $schedArgs = @('/SC','DAILY','/ST', $At)
        Write-Host "[register] daily at $At (local)"
    }
    else {
        Write-Host "ERROR: must supply one of -At, -EveryMinutes, -Once" -ForegroundColor Red
        exit 1
    }

    $args = @('/Create','/F','/TN', $Name,'/TR', $tr) + $schedArgs
    & schtasks.exe @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "schtasks failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "[register] OK - task '$Name' registered." -ForegroundColor Green
    Write-Host "  Inspect:  schtasks /Query /TN $Name /V /FO LIST"
    Write-Host "  Run now:  .\local_scheduler.ps1 run -Name '$Name'"
}

function Action-List {
    Write-Host "[list] AI-Trader local tasks:" -ForegroundColor Cyan
    $out = & schtasks.exe /Query /FO CSV /NH 2>$null
    if (-not $out) { Write-Host "  (none)"; return }
    $out | Where-Object { $_ -match 'AI-Trader' } | ForEach-Object {
        $cols = $_ -split '","'
        $name = $cols[0].TrimStart('"')
        $next = $cols[1]
        $stat = $cols[2].TrimEnd('"')
        Write-Host ("  {0,-40} next: {1,-25} status: {2}" -f $name, $next, $stat)
    }
}

function Action-Unregister {
    & schtasks.exe /Delete /TN $Name /F
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[unregister] OK - '$Name' removed." -ForegroundColor Green
    }
}

function Action-Run {
    & schtasks.exe /Run /TN $Name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[run] OK - triggered '$Name' now." -ForegroundColor Green
    }
}

switch ($Action) {
    'register'   { Action-Register }
    'list'       { Action-List }
    'unregister' { Action-Unregister }
    'run'        { Action-Run }
}
