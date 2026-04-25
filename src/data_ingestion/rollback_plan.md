# Rollback & Fix Plan

This document contains the exact steps to roll back the broken dashboard layout to the stable stacked-card design, fix the ML prediction crash, and fully translate the bot to English.

## 1. Fix the Fatal ML Crash & Orchestration in `src/main.py`
**The Bug:** The bot crashes after adding a new coin because it forgets to load the 1-minute features for the scalping model. Also, data wasn't downloading if the Binance ping failed.
**The Fix:** 
- Change `self.scalping_predictor = MLPredictor(model_filename='scalping_model.joblib')` 
  to `self.scalping_predictor = MLPredictor(model_filename='scalping_model.joblib', model_type='scalping')` inside `check_and_prepare_new_symbols()`.
- Move `self.check_and_prepare_new_symbols()` to the very beginning of the `run()` function.
- Translate all `logger.info` and `logger.error` messages to English.

## 2. Fix Aggressive Process Killing in Restart Scripts
**The Bug:** `python.exe` processes get stuck in the background and hold onto the port, preventing the new code from loading.
**The Fix:**
- In `stop_all.bat`: Replace the powershell kill command with `taskkill /F /IM python.exe /T >nul 2>&1`.
- In `restart_all.ps1`: Replace the WMI kill command with `taskkill /F /IM python.exe /T 2>&1 | Out-Null`.

## 3. Translate API Responses in `src/dashboard/app.py`
**The Fix:** Update fallback JSON responses to English.
- Change `"Нет данных"` to `"No data"`.
- Change `"Логов пока нет..."` to `"No logs yet..."`.
- Translate the AI Assistant system prompts and default error responses to English.

## 4. Full HTML Rollback (`src/dashboard/templates/index.html`)
**The Fix:** Completely replace `index.html` to remove the broken sidebar grid. 
Use the classic single-column stacked layout where every section is a collapsible card:
1. Main Summary & Controls (Balances, Risk, Signal, Portfolio)
2. Watchlist Management & Market Data (Added as a standard card)
3. Interactive Chart & AI HUD
4. Strategy Performance
5. Active Trades
6. Trade History
7. Gemini AI Assistant Chat
8. Terminal Actions Log

*Note: The full HTML code for this clean, English, stacked layout is ready to be pasted tomorrow.*

## Execution Steps for Tomorrow:
1. Apply the code changes to the respective files.
2. Double-click `stop_all.bat` to forcefully clear any stuck background processes.
3. Double-click `restart_all.bat`.
4. Once the terminal shows the bot is running, go to the browser and press **Ctrl + F5** (or Cmd + Shift + R) to hard-refresh the layout cache.