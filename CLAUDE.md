# Claude Code Instructions

## Approval Gate (MANDATORY)

Before writing ANY code, always present a written implementation plan and wait for explicit user approval.
- Describe: what files change, what new files are created, the approach taken, and any key decisions.
- Do NOT start implementing until the user says "approved", "go ahead", "yes", or equivalent.
- **Answering clarifying questions is NOT approval.** When the user replies to a plan with answers to "(1) (2) (3)" style questions, that's clarification, not authorization. Re-state the resolved plan and ASK for explicit approval before any tool call that creates/edits files. Only "approved", "go ahead", "yes", "implement", or equivalent counts.
- **All sub-phases of an approved plan ARE auto-approved.** Once approval is given for a multi-layer plan, every layer/sub-step inside that plan can proceed without re-asking. The gate is on the PLAN boundary, not each step within it.
- If the user adds NEW scope mid-conversation (new requirements, new layers), treat the expanded plan as un-approved and re-confirm. Original approval only covers original scope.
- **DOUBLE-ASK before plan execution.** After receiving plan approval ("yes/go/approved"), DO NOT immediately start writing code. Restate the plan succinctly (one paragraph) and ASK ONE MORE TIME: "Confirm to proceed?". Wait for the second explicit confirmation. Only THEN make the first tool call that creates/edits files. Sub-steps inside a double-confirmed scope are auto-approved (don't ask before every micro-step). Reason: 2026-05-10 — interpreted a "correction" message as approval and started implementing Option B, while the user actually meant clarification of the requirement; user had to ask "did you get my reply?" to catch the over-eager start. The first-pass approval is necessary but not sufficient — second confirmation is the gate.

## No Guessing (MANDATORY)

When asked a factual question about state ("is X working?", "is it correct?", "what's happening?"):
- **TEST first** — query the actual system (logs, processes, HTTP endpoints, files). Make the test specific enough that the result distinguishes between hypotheses.
- **If you cannot test or aren't sure**: ASK the user for the missing detail before answering.
- **Never substitute speculation for evidence.** Words like "probably", "most likely", "should be", "likely cause" without a test backing them = banned.
- When you DO answer, lead with the test result, not the conclusion. e.g. "nvidia-smi shows 0% util + no python in compute apps → GPU is not being used" — NOT "the trainer is probably still in data prep".
- Reason: 2026-05-10 — speculated repeatedly during a TFT smoke test about whether GPU was being used. User correctly called it out: stop guessing, test or ask.

## Workflow Rules

- Run `restart_all.ps1` after every completed task so the live bot and dashboard always reflect latest code.
- After every code change, add/update test assertions in `tests/test_dashboard.py` and confirm 0 failures.
- Always create a git commit of the current state before starting any new implementation phase, especially multi-file refactors. Don't bundle unrelated fixes into a migration commit.
- Gemini model fallback chain: `gemini-3.1-pro-preview` must be first. Update if a newer model releases.
- **Bash / PowerShell / curl commands are pre-approved.** Don't ask permission to run shell commands — read-only probes, file inspection, log tails, process listings, port scans, training/data triggers, and similar diagnostic or operational commands all run without confirmation. Only ask first for: destructive ops (rm -rf, force-push, drop table), things that touch shared state (publishing to remotes, sending external messages), or actions outside this repo. In short: act like the operator already typed "yes" for the safe stuff.

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
