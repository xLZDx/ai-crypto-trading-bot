# AI Session Notes

## Core Constraints
- **Execution Style**: Work on tasks strictly one by one. No parallelism or concurrent execution to avoid freezing.

## Session History & Context
- **Current Project State**: AI Trading Assistance. Contains modules for data ingestion (Binance), analysis (Elliott Waves, Risk Management), engine (Order Manager, Trade Tracker), and a dashboard.
- **Last Action**: Established the sequential task execution rule and created this notes file to track session history going forward.

## Debugging & Troubleshooting Protocol
- **Stop Guessing**: When a crash or complex bug occurs, do not guess the solution.
- **Use Debug Server**: Ask the user to attach their IDE (VS Code/PyCharm) to the live debug server on port `5678` to trace the exact failure point.
- **Collect Clear Logs**: Request the exact, full stack trace from the terminal or `logs/trading.log` (which is now caught globally in `main.py`) before writing any fix.

## Pending Tasks (Rollback & Fix Plan)
- **Goal**: Roll back dashboard layout to the stable stacked-card design, fix the ML prediction crash (`scalping_model` model_type initialization), and fully translate the bot to English.
- **Files to modify next**:
  1. `src/main.py`: Fix `MLPredictor` scalping init and translate logs.
  2. `stop_all.bat` & `restart_all.ps1`: Use `taskkill /F /IM python.exe /T` to prevent stuck processes.
  3. `src/dashboard/app.py`: Translate API responses to English.
  4. `src/dashboard/templates/index.html`: Completely replace with the classic single-column stacked layout (in English).
- **Next Steps Execution**: Apply code changes, run `stop_all.bat`, run `restart_all.bat`, and hard-refresh the browser (Ctrl+F5).

*Note: Please update this file at the end of future sessions or major milestones to easily pick up where we left off.*