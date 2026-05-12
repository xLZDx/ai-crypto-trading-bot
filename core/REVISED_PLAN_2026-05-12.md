# Revised Stabilization Plan — 2026-05-12

**Status:** APPROVED in principle (operator 2026-05-12); held pending consolidated meta-plan combining this + Sprint 0/0a/0b/0c + analytic phase.
**Supersedes:** N/A (new plan).
**Inputs:** Codebase audit (code-explorer), 3 specialist reviews (security-reviewer, architect, python-reviewer), 2026-05-12.

This document is the **canonical revised plan** from the 2026-05-12 audit. It will be folded into a larger consolidated meta-plan that also incorporates:
- `TECH_IMPLEMENTATION_PLAN_2026-05-10.md` (Sprint 0 + 0a/0b/0c + analytic)
- `COMPETITIVE_ASSESSMENT_2026-05-10_v2.md` §11 (personal-use-pruned items)

The work below is **Stabilization Sequence** — must complete before Sprint 0 begins.

---

## Phase ordering (corrected)

```
A  Security hardening (~2 d) ──→  C  State + agent consolidation (~2-3 d)
                                  ↓
                                  B  ML correctness (~1 d + overnight)
                                  ↓
                                  D  Trainer cleanup (~0.5 d)
                                  ↓
                                  E  Performance (~1 d)         ─┐ parallel
                                  F  Test rebuild (~2 d)         ─┘
                                  ↓
                                  G  Risk mgmt features (~1 d)
                                  ↓
                                  [Stabilization complete → Sprint 0 begins]
```

**Total:** ~9-10 working days serial / ~7 days with F parallel.

---

## Phase A — Security hardening (~2 days)

Twelve tasks across CRITICAL/HIGH/MEDIUM/LOW severity. Three CRITICAL items were missed in the original audit and only caught by the specialist security review.

### CRITICAL
1. **A1** — Add `@require_api_key` to all 50+ unprotected routes, including 3 NEW CRITICAL ones missed by the original audit:
   - `/api/scheduler/*` (5 routes, [app.py:6305-6399](src/dashboard/app.py#L6305)) — invokes `schtasks.exe`, OS-level persistence
   - `/api/cluster/worker_restart` ([app.py:5925](src/dashboard/app.py#L5925)) — write-side SSRF
   - `/api/cluster/register` + `/api/cluster/task_update` ([app.py:5807,5816](src/dashboard/app.py#L5807)) — cluster poisoning
   - `/api/control/trade_mode` GET ([app.py:6467](src/dashboard/app.py#L6467))
   - Full enumerated list in audit transcript
2. **A2** — Bind monitor:5001, worker:7701, orchestrator:7700, control_plane:8100 to `127.0.0.1` (currently `0.0.0.0`)
3. **A3** — IP allowlist + shared API key on `/api/cluster/worker_restart`

### HIGH
4. **A4** — Replace `/api/db/query` with per-table parameterized read endpoints; or gate with auth + LIMIT + timeout ([app.py:2517](src/dashboard/app.py#L2517))
5. **A5** — Parameterize all DuckDB f-string queries ([train_tft_model.py:193](src/engine/train_tft_model.py#L193), [data_lens.py:131](src/analytics/data_lens.py#L131), [realtime_db_writer.py:202](src/data_ingestion/realtime_db_writer.py#L202), [parquet_store.py:180](src/database/parquet_store.py#L180))
6. **A6** — ZMQ HMAC envelope + drop pickle fallback ([data_bus.py:55](src/transport/data_bus.py#L55))
7. **A7** — `torch.load(weights_only=True)` on all 3 sites
8. **A8** — HMAC integrity check on **ALL 12 model load sites** (3 torch + 9 joblib) — write `models/manifest.json` (path → SHA-256 + HMAC), verify before load. 9 joblib sites: backtester.py:387, futures_agent.py:52, ml_predictor.py:40, scalping_agent.py:55, meta_labeler.py:72, regime_classifier.py:114, spot_agent.py:52,55, train_meta_labeler.py:83-85

### MEDIUM
9. **A9** — Short-lived session token instead of raw API key in HTML template ([app.py:121](src/dashboard/app.py#L121))
10. **A10** — `flask-cors` restricted to `127.0.0.1`; `flask-limiter` on `/api/chat`
11. **A11** — Migrate unique endpoints from `src/server/control_plane.py :8100` into `app.py` under auth; delete `control_plane.py` + `launch_fastapi.ps1`; update `restart_all.ps1` + `error_monitor.py`

### LOW
12. **A12** — Replace raw `open()` writes on lock/state files with `safe_json.write_json()` ([train_all_models.py:139,171](src/training/train_all_models.py#L139))

---

## Phase C — State persistence + agent consolidation (~2-3 days)

Architect flagged this MUST run before Phase B retrain (orchestrator state is volatile until persisted).

1. **C1** — Persist orchestrator `_workers`/`_tasks`/`_queue` to `data/orchestrator_state.json` via `safe_json`; reload on startup ([orchestrator.py:76](src/training/distributed/orchestrator.py#L76))
2. **C2** — Task dedup by `(model_key, timeframe)` ([orchestrator.py:145](src/training/distributed/orchestrator.py#L145))
3. **C3** — Delete `src/agents/master_agent.py` + `trainer_example_agent.py` + `tests/test_agents.py` + `src/agents/__init__.py` + README; canonicalize `src/orchestration/master_agent.py` (387 lines, real zombie healer)
4. **C4** — `threading.RLock` on `ParquetStore` ([parquet_store.py:98](src/database/parquet_store.py#L98))
5. **C5** — Phase 100b — `tf='all'` + pipeline orchestrator route through cluster
6. **C6** — Phase 100c — delete `_TrainingScheduler` + Phase 97c orphan detection (after 100b stable for 1 cycle)
7. **C7** — Trim `data/agent_status.json` history array to `max_history=100`

---

## Phase B — ML correctness (~1 day + overnight retrain)

Python-reviewer flagged that B3 needs 4 guards without which the fix is worse than current hardcoded values.

0. **B0** — Snapshot `models/` to `models/archive/2026-05-XX/` before any retrain (30+ dirty meta files in git status)
1. **B1** — Rewrite `PurgedKFold` as true walk-forward (start fold=1, `train_end = max(0, test_start − embargo_size)`); concrete code pattern provided by python-reviewer ([purged_kfold.py:35](src/utils/purged_kfold.py#L35))
2. **B2** — Three non-overlapping splits: train 0–70 / cal 70–85 / test 85–100 ([train_model.py:220](src/engine/train_model.py#L220))
3. **B3** — HP from `training_rules.json` with 4 guards:
   - Loader WARNS on missing keys (incl. `learning_rate` which is absent from JSON)
   - Document that JSON `max_depth=8` + `n_estimators=100` differs from current `max_depth=6, n_estimators=500`
   - Schema validator at load (required keys + numeric range)
   - Version/checksum JSON alongside saved model artifact
4. **B4** — Explicit WARN when news sentiment CSV missing ([train_model.py:124](src/engine/train_model.py#L124))
5. **B5** — Retrain all 22 models on corrected pipeline; commit new `_meta.json`

---

## Phase D — Trainer cleanup (~0.5 day) — INVERTED

Architect caught: `src/engine/trainers/` are **thin wrappers that import FROM** `src/engine/train_*_model.py`. Deleting top-level files would break [worker.py:276-284](src/training/distributed/worker.py#L276) cluster dispatch, train_all_models, dashboard. ORIGINAL plan was wrong.

1. **D1** — DELETE the WRAPPERS in `src/engine/trainers/`, NOT the top-level files
2. **D2** — Delete `src/engine/train_model_v2.py` + update [tests/test_dashboard.py:1243-1244](tests/test_dashboard.py#L1243) + [tests/test_phase10.py:74-76](tests/test_phase10.py#L74) in same commit
3. **D3** — Delete `src/tools/binance_archive_downloader.py` (no production importer); fix [app.py:524](src/dashboard/app.py#L524) import to use `src/data_ingestion/` version

---

## Phase E — Performance (~1 day)

1. **E1** — Single `psutil.process_iter()` per monitor request → PID dict ([app.py:656](src/dashboard/app.py#L656))
2. **E2** — Persisted size manifest for parquet tree, replace 48 GB `rglob().sum()` ([app.py:868](src/dashboard/app.py#L868))
3. **E3** — `/api/db/query` (kept as authenticated read endpoint) — enforce `LIMIT 10000` + query timeout
4. **E4** — Portfolio context TTL cache (10s) for `/api/chat` ([app.py:229](src/dashboard/app.py#L229))
5. **E5** — Close log file handles in supervisor on subprocess exit ([master_agent.py:162](src/orchestration/master_agent.py#L162))

---

## Phase F — Test rebuild (~2 days, parallel with D/E)

1. **F1** — `tests/test_safe_json.py` — concurrent writers, lock contention, atomic rename
2. **F2** — `tests/test_purged_kfold.py` — anti-leakage guarantee (no test index in any train fold)
3. **F3** — `tests/test_dashboard_api.py` — `app.test_client()` round-trips for 15 critical endpoints
4. **F4** — `tests/test_orchestrator.py` — submit, dedup, state-persistence-roundtrip, worker assignment
5. **F5** — `tests/test_parquet_store.py` — ingest small fixture, query, threading
6. **F6** — `tests/test_model_integrity.py` (NEW from A8) — HMAC manifest verify + reject on tamper
7. **F7** — Reduce `tests/test_dashboard.py` string-matches to supplementary smoke checks only

---

## Phase G — Risk management features (~1 day)

Files A/B/C from operator proposal, with python-reviewer's enhanced signatures.

1. **G1** — Module-level `calc_liquidation_price(entry, leverage, side, maint_margin_rate=0.005, taker_fee_rate=0.0004, accumulated_funding=0.0, margin_type="isolated")` — assert on `margin_type != "isolated"` (cross-margin formula is different) ([src/analysis/risk_manager.py](src/analysis/risk_manager.py))
2. **G2** — `HullRiskManager.size_from_stop_distance(...) → float` returning **position notional in quote currency** (explicit docstring); type hints on all `HullRiskManager` methods
3. **G3** — `src/analysis/live_funding.py` with shared ccxt instance, `threading.Lock`-protected TTL cache (avoid TOCTOU), ccxt rate limiter, fail-closed
4. **G4** — Wire G1+G2+G3 into `futures_agent._on_signal()` ([futures_agent.py:66](src/engine/agents/futures_agent.py#L66))
5. **G5** — Audit `order_manager.py` for `HullRiskManager` wiring
6. **G6** — `tests/test_risk.py` — liquidation correctness (long/short/with-fee/with-funding), size sanity, TTL cache thread-safety, futures_agent rejection-on-funding

---

## What's explicitly NOT in this plan (deferred to consolidated meta-plan)

- Sprint 0 §S0-1 to §S0-6 — validation rigor harness, model bake-off, kill switch, exec quality dashboard, calibration audit, cut list
- Sprint 0a M1/M2/M3/M4 — position caps, correlation monitor, leverage cap, tick circuit breaker
- Sprint 0b O1/O2/O3/O4/O5 — backups, offsite, reconciler, network SAFE MODE, UPS runbook
- Sprint 0c C1/C2/C3 — multi-exchange, cold storage, exchange health
- §S0.5 analytic phase — execute cut list
- Telegram OUTPUT — operator rejected 2026-05-12

These are the *next* sequence after stabilization completes.

---

## Open question

`restart_all.ps1` is the supervisor's supervisor for `master_agent.py`. Acceptable for personal-use single-laptop, or add a Windows scheduled task? Operator pending decision.
