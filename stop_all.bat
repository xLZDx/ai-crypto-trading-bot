@echo off
echo Stopping AI Trading Assistance...

:: 1. Terminate the actual Python processes running main.py and app.py
echo Stopping Python processes...
wmic process where "name='python.exe' and (commandline like '%%src\\main.py%%' or commandline like '%%src\\dashboard\\app.py%%' or commandline like '%%server.py%%')" call terminate >nul 2>&1

:: 2. Close the console windows
echo Closing terminal windows...
taskkill /FI "WINDOWTITLE eq AI Trading Bot*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq AI Trading Dashboard*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq AI Trading MCP Server*" /T /F >nul 2>&1

echo.
echo All processes have been stopped!
pause
