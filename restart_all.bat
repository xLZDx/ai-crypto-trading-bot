@echo off
title AI Trading Assistance — CLEAN RESTART
cd /d "%~dp0"

echo ============================================================
echo   AI TRADING ASSISTANCE — CLEAN RESTART
echo ============================================================
echo.
echo Step 1/2 — STOP every managed process (bot, dashboard, monitor,
echo            training, realtime DB writer, data orchestrator,
echo            watchlist downloader). Strays are swept too.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_all.ps1"
if errorlevel 1 (
    echo.
    echo WARNING: stop_all.ps1 returned a non-zero exit code, continuing anyway.
    echo.
)

echo.
echo Waiting 3 seconds for sockets / file locks to release ...
timeout /t 3 /nobreak >nul

echo.
echo Step 2/2 — START the full pipeline.
echo            Progress is printed below in real time. This window stays open.
echo.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0restart_all.ps1"
