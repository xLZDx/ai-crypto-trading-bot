# Claude Code Instructions

## Approval Gate (MANDATORY)

Before writing ANY code, always present a written implementation plan and wait for explicit user approval.
- Describe: what files change, what new files are created, the approach taken, and any key decisions.
- Do NOT start implementing until the user says "approved", "go ahead", "yes", or equivalent.
- Once a plan is approved, implementation and automated testing can proceed without per-step approval.

## Workflow Rules

- Run `restart_all.ps1` after every completed task so the live bot and dashboard always reflect latest code.
- After every code change, add/update test assertions in `tests/test_dashboard.py` and confirm 0 failures.
- Always create a git commit of the current state before starting any new implementation phase, especially multi-file refactors. Don't bundle unrelated fixes into a migration commit.
- Gemini model fallback chain: `gemini-3.1-pro-preview` must be first. Update if a newer model releases.

## Project Context

- Working directory: `D:\test 2\AI trading assistance`
- Python venv: `venv\`
- Main entry: `src/main.py` (trading engine), `src/dashboard/app.py` (Flask UI, port 5000)
- DB: ParquetClient (DuckDB + partitioned Parquet on `data/db/`) — file-based, no daemon. Replaces QuestDB after the Phase 1–5 migration (commits 43db156…). Historical OHLCV (48 GB) still served by `parquet_store.py` from `data/parquet/`.
- All file I/O uses `src/utils/safe_json.py` (filelock atomic writes)
- All constants centralized in `src/utils/config.py`
- Execution is strictly sequential — no parallelism inside the bot loop
- Testnet by default — do not switch to Mainnet without explicit user instruction
- All cache/temp data must go to D: drive, never C:
- Use `pip install --no-cache-dir ...` so the pip wheel cache does not accumulate on C:
- DuckDB connections must set `temp_directory='D:/.../data/cache/duckdb_temp'` (already done in `src/database/parquet_store.py`)
