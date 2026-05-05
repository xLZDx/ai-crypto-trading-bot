@echo off
title AI Trading Assistance — START ALL
cd /d "%~dp0"

echo ============================================================
echo   AI TRADING ASSISTANCE — START ALL
echo ============================================================
echo.
echo Delegating to restart_all.ps1 (idempotent, full pipeline).
echo Progress is printed below in real time.
echo.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0restart_all.ps1"
