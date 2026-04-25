# AI Session Notes

## Core Constraints
- **Execution Style**: Work on tasks strictly one by one. No parallelism or concurrent execution to avoid freezing.
- **Version Control / Rollback Rule**: Always document and track changes being made during the session. Ensure that we always have a clear path to roll back to the last known working version if a new feature breaks the system.

## Session History & Context
- **Current Project State**: AI Trading Assistance. Contains modules for data ingestion (Binance), analysis (Elliott Waves, Risk Management), engine (Order Manager, Trade Tracker), and a dashboard.
- **Last Action**: Successfully rolled back the dashboard to a stable stacked layout, fixed ML prediction crashes, fully translated the codebase to English, and enforced a strict rollback tracking rule.

## Debugging & Troubleshooting Protocol
- **Stop Guessing**: When a crash or complex bug occurs, do not guess the solution.
- **Use Debug Server**: Ask the user to attach their IDE (VS Code/PyCharm) to the live debug server on port `5678` to trace the exact failure point.
- **Collect Clear Logs**: Request the exact, full stack trace from the terminal or `logs/trading.log` (which is now caught globally in `main.py`) before writing any fix.

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

*Note: Please update this file at the end of future sessions or major milestones to easily pick up where we left off.*