> **Inherits global rules from `D:\test 2\CLAUDE.md`** — approval gate, no-guessing, regression tests, git lifecycle (including todo-in-commits), shell pre-approval, D:-drive-only disk policy. Read that file too.

# AI Trading Assistance — Project Context

## Layout
- Working directory: `D:\test 2\AI trading assistance`
- Python venv: `venv\`
- Main entry: `src/main.py` (trading engine)
- Dashboard: `src/dashboard/app.py` (Flask UI, port 5000)
- Restart everything: `restart_all.ps1`

## Architecture
- **DB:** ParquetClient (DuckDB + partitioned Parquet on `data/db/`). File-based, no daemon. Replaces QuestDB after Phase 1–5 migration.
- **Historical OHLCV:** 48 GB served by `parquet_store.py` from `data/parquet/`.
- All file I/O on JSON state goes through `src/utils/safe_json.py` (filelock atomic writes).
- Constants centralized in `src/utils/config.py`.
- Execution inside the bot loop is **strictly sequential** — no parallelism.

## Defaults
- **Testnet by default** — do NOT switch to Mainnet without explicit user instruction.
- DuckDB connections must set `temp_directory='D:/test 2/AI trading assistance/data/cache/duckdb_temp'` (done in `src/database/parquet_store.py`).
- Gemini model fallback chain: `gemini-3.1-pro-preview` must be first. Update when a newer model releases.

## Test path
- Canonical regression suite: `tests/test_dashboard.py`. 0 failures required before push.
- After every code change, add/update assertions and verify 0 failures.

## Per-task workflow
- Run `restart_all.ps1` after every completed task so the live bot and dashboard reflect latest code.

## Project plan & state
- Outstanding-work plan: `PLAN_2026_05_08_outstanding.md`.
- Phase 100 (cluster-routed training): `core/PHASE_100_CLUSTER_ROUTED_TRAINING.md`.
- Tech implementation plan: `TECH_IMPLEMENTATION_PLAN_2026-05-10.md`.
- Sprint 1A (per-model agents + KPI): `core/SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md`.
- Worker laptop "Ivan" launcher: `C:\ai-worker\restart_workers.ps1` (this is the harness-imposed bridge — see global D:-drive policy; it must stay on C: because of the worker OS install, but logs/data live on D:).
- Cluster orchestrator port: 7700.
- Model × TF coverage matrix: `data/training_rules.json` — read on every training startup.
