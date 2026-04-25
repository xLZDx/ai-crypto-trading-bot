# AI Session Notes

## Core Constraints
- **Execution Style**: Work on tasks strictly one by one. No parallelism or concurrent execution to avoid freezing.
- **Version Control / Rollback Rule**: Always document and track changes being made during the session. Ensure that we always have a clear path to roll back to the last known working version if a new feature breaks the system.
- **Checkpoint Commit Rule**: After each completed change or milestone, automatically create a local git commit without asking for approval so the project can be restored to a previous checkpoint later.

## Session History & Context
- **Current Project State**: AI Trading Assistance. Contains modules for data ingestion (Binance), analysis (Elliott Waves, Risk Management), engine (Order Manager, Trade Tracker), and a dashboard.
- **Last Action**: Successfully rolled back the dashboard to a stable stacked layout, fixed ML prediction crashes, fully translated the codebase to English, and enforced a strict rollback tracking rule.

## Debugging & Troubleshooting Protocol
- **Stop Guessing**: When a crash or complex bug occurs, do not guess the solution.
- **Plan First**: Before implementing any change, prepare a clear written plan/todo and show it to the user for review, even if bypass approval is available.
- **Todo List First**: Always show the todo list before implementation and allow the user to edit it before approval.
- **Use Debug Server**: Ask the user to attach their IDE (VS Code/PyCharm) to the live debug server on port `5678` to trace the exact failure point.
- **Collect Clear Logs**: Request the exact, full stack trace from the terminal or `logs/trading.log` (which is now caught globally in `main.py`) before writing any fix.

## User-Requested Workflow Rules
- **Completion Report Required**: When a task is completed, always show exactly what was completed, what changed, what was added, and what was missed relative to the todo list.
- **Rollback-Friendly Changes**: Keep changes grouped into checkpoint commits so each implementation stage can be rolled back independently.
- **Automatic Commit Policy**: After implementation work is finished, commit all completed changes automatically so the user can roll back far back if needed.
- **Approval Before Work**: Present the todo list first, then ask for approval, then proceed with implementation.
- **Editable Plan**: Let the user modify the todo list before starting implementation.

## Completed (2026-04-25 Session)
- Full codebase security & quality review and fixes (see summary below)

## Pending Tasks
- **Goal**: Integrate Telegram Bot for live trade notifications.

## New Files Added (2026-04-25)
- `src/utils/__init__.py` — utils package
- `src/utils/safe_json.py` — atomic JSON read/write with filelock (use everywhere instead of `open()`)
- `src/utils/config.py` — all magic numbers/constants centralized here
- `src/analysis/feature_engineering.py` — shared ML indicator functions (RSI, MACD, ATR, ADX, BB, ROC)

## New Dependencies Added (requirements.txt)
- `filelock>=3.12.0` — for safe_json file locking
- `defusedxml>=0.7.1` — for sentiment.py XML parsing

## Dashboard Auth
- Set `DASHBOARD_API_KEY=<your_secret>` in `.env`
- All `/api/*` routes require header `X-API-Key: <your_secret>`
- If key not set, access is unprotected (with a startup warning)

## Completed This Session (2026-04-25)
- Identified the live dashboard processes and confirmed there were two active `src/dashboard/app.py` instances.
- Created a safe PowerShell restart script for the dashboard to avoid command-line quoting issues.
- Restarted the dashboard service cleanly.
- Mapped the bot architecture and entry points:
  - `src/main.py` is the live trading engine.
  - `src/dashboard/app.py` is the Flask dashboard.
  - `src/analysis/` provides analytics, ML, sentiment, and risk logic.
  - `src/engine/` provides execution, trade tracking, and Gemini veto logic.
- Confirmed the current dashboard route set and data sources.
- Documented the last implemented refactor/hardening pass and its files.
- Recorded the pending next feature: Telegram bot notifications.

## Session Rule
- After every completed implementation, append the finished checklist items and any relevant notes here before closing the task.
- Maintain this file as the source of truth for workflow preferences, rollback checkpoints, and completion reporting.

*Note: Please update this file at the end of future sessions or major milestones to easily pick up where we left off.*
