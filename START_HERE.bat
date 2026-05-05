@echo off
title AI Trading Assistance — START
cd /d "%~dp0"

echo ============================================================
echo   AI TRADING ASSISTANCE — START
echo   This window stays open and shows full progress.
echo ============================================================
echo.
echo Calling restart_all.ps1 ...
echo.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0restart_all.ps1"
