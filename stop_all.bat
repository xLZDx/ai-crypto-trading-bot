@echo off
title AI Trading Assistance — STOP
cd /d "%~dp0"

echo ============================================================
echo   AI TRADING ASSISTANCE — STOP ALL
echo ============================================================
echo.
echo Calling stop_all.ps1 (kills bot, dashboard, monitor, training,
echo  realtime DB writer, data orchestrator, watchlist downloader).
echo Progress is printed below in real time.
echo.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0stop_all.ps1"
