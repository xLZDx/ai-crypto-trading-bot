$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AI TRADER: POWERSHELL AUTO-SETUP"
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n[1/5] Terminating old background processes..." -ForegroundColor Yellow
# Read saved PIDs from last run and kill them (no WMI / no hang)
$pidFile = Join-Path $PSScriptRoot 'data\process_ids.json'
if (Test-Path $pidFile) {
    try {
        $saved = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($p in @($saved.bot, $saved.dash, $saved.mcp)) {
            if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
        }
    } catch {}
}
# Fallback: kill by window title (fast — uses Get-Process, no WMI)
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -match "AI Trading Bot|AI Trading Dashboard|AI Trading MCP Server" } |
    Stop-Process -Force -ErrorAction SilentlyContinue
# Last resort: taskkill on any python window matching our title pattern
& taskkill /F /FI "WINDOWTITLE eq AI Trading Bot*"    2>$null | Out-Null
& taskkill /F /FI "WINDOWTITLE eq AI Trading Dashboard*" 2>$null | Out-Null
& taskkill /F /FI "WINDOWTITLE eq AI Trading MCP Server*" 2>$null | Out-Null

# Wait for terminated processes to fully release DLLs before spawning new ones
Start-Sleep -Seconds 3

Write-Host "`n[2/5] Setting up Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creating new Python virtual environment in 'venv'..." -ForegroundColor Magenta
    python -m venv venv
}

Write-Host "`n[3/5] Installing all missing libraries..." -ForegroundColor Yellow
$pythonPath = ".\venv\Scripts\python.exe"
$pipPath = ".\venv\Scripts\pip.exe"
& $pythonPath -m pip install --quiet --upgrade pip
& $pipPath install --quiet -r requirements.txt
& $pipPath install --quiet websockets vaderSentiment ccxt python-dotenv flask pandas scikit-learn joblib mcp google-genai youtube-transcript-api beautifulsoup4 requests debugpy
Write-Host "Libraries installed successfully." -ForegroundColor Green

Write-Host "`n[4/5] Launching ML Training in background (non-blocking)..." -ForegroundColor Yellow
$trainCmd = "-NoExit -Command `"Set-Location '$PSScriptRoot'; `$host.UI.RawUI.WindowTitle = 'ML Training [4/4 models]'; .\venv\Scripts\Activate.ps1; python src\engine\train_all_models.py; Write-Host 'All models trained. You may close this window.' -ForegroundColor Green; Start-Sleep 5`""
Start-Process powershell -ArgumentList $trainCmd
Start-Sleep -Seconds 2

Write-Host "`n[5/5] Launching the system..." -ForegroundColor Yellow
$botCmd = "-NoExit -Command `"Set-Location '$PSScriptRoot'; `$host.UI.RawUI.WindowTitle = 'AI Trading Bot'; .\venv\Scripts\Activate.ps1; python -m debugpy --listen 0.0.0.0:5678 src\main.py`""
$dashCmd = "-NoExit -Command `"Set-Location '$PSScriptRoot'; `$host.UI.RawUI.WindowTitle = 'AI Trading Dashboard'; .\venv\Scripts\Activate.ps1; python src\dashboard\app.py`""
$mcpCmd = "-NoExit -Command `"Set-Location '$PSScriptRoot'; `$host.UI.RawUI.WindowTitle = 'AI Trading MCP Server'; .\venv\Scripts\Activate.ps1; python src\mcp_server\server.py`""

# Stagger launches — simultaneous spawns cause 0xc0000142 DLL init failure
$procDash = Start-Process powershell -ArgumentList $dashCmd -PassThru
Start-Sleep -Seconds 2
$procBot  = Start-Process powershell -ArgumentList $botCmd  -PassThru
Start-Sleep -Seconds 2
$procMcp  = Start-Process powershell -ArgumentList $mcpCmd  -PassThru

# Save PIDs so next restart can kill exactly these processes without WMI
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'data') | Out-Null
@{ bot = $procBot.Id; dash = $procDash.Id; mcp = $procMcp.Id } |
    ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'data\process_ids.json')

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "   ALL PROCESSES STARTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green