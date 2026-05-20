# UNIFIED MASTER PLAN — AI Trading Bot — 2026-05-20
**Agent review completed 2026-05-20** by architect · planner · python-reviewer · security-reviewer.
All findings integrated below.
**Operator review + priority restructure completed 2026-05-20** (P0/P1/P2 gate format).

**Supersedes ALL prior plans:**
- `MASTER_PLAN_2026-05-20.md`
- `CONSOLIDATED_META_PLAN_2026-05-12.md`
- `PLAN_DEGRADATION_MONITORING_2026-05-16.md`
- `PLAN_POST_PRODUCTION_TUNING.md`
- `PLAN_VPS_CLEAN_SLATE.md` / `PLAN_VPS_CLEAN_SLATE_RU.md`
- `PLAN_2026_05_08_outstanding.md` / `PLAN_2026_05_07.md` / `PLAN_2026_05_07_followup.md`
- `core/CONSOLIDATED_META_PLAN_2026-05-12.md`
- `core/SPRINT_1A_PER_MODEL_AGENTS_AND_KPI.md`
- `core/REVISED_PLAN_2026-05-12.md`

**Branch:** `dev/vps-clean-slate` → merge to `main` after all GATE 0 + 1 + 2 pass
**VPS:** 5.104.81.27 (Contabo Tokyo, 400 GB SSD, 24 GB RAM, Ubuntu 24.04)
**Bot purpose:** personal-use only — capital preservation + profit for the operator

---

## Status Legend
- ✅ Done | 🔄 In progress | ❌ Pending | ➡️ Blocked | 🚫 Requires separate GO

---

## PRIORITY LEVELS

> **Operator directive (2026-05-20):** "Всё, что связано с retrain, dashboard beauty, analytics, taxonomy и advanced alpha, должно ждать, пока система не станет fail-closed, authenticated, signed, observable и способной остановить торговлю сама."

| Level | Definition | Blocks |
|---|---|---|
| **P0** | Cannot merge to `main` / cannot touch real cash | everything |
| **P1** | Must complete before E3 retrain GO | E3, real cash |
| **P2** | After baseline production hardening | — |

---

## GATE STRUCTURE

```
GATE 0-A  Unicode / encoding safety        (Z1)               P0  Day 0
GATE 0-B  Auth + surface hardening         (A3,I1,I2,I5,I6)  P0  Day 0–1
────────────────────────────────────────────────────────────────────────
GATE 1    Model integrity: sign + verify   (Z2,I4)            P0  Week 1
GATE 2    Live risk stop                   (C1,C2,N5,Z3,N8)   P0  Week 1–2
GATE 3    Futures / risk math              (C7,I7,C9,N6,C3,C6)P1  Week 2–3
────────────────────────────────────────────────────────────────────────
P1 block  Before E3 retrain               (N1,N2,J1–J5,I3,
                                           B10,I8,N3,N7)      P1  Week 2–4
────────────────────────────────────────────────────────────────────────
E3 GO     (requires ALL P0 + P1 complete) ← separate ГО
────────────────────────────────────────────────────────────────────────
P2 block  After baseline hardening        (B*,D*,F*,G*,H*,K*) P2  later
```

---

## COMPLETED (do not re-implement)

| # | Item | Commit |
|---|---|---|
| 1 | VPS firewall (ufw 22/5000), fail2ban, SSH key-only | prior session |
| 2 | `dev/vps-clean-slate` branch pushed | prior session |
| 3 | WebSocket ping_timeout=60 / close_timeout=15 | c24b789 |
| 4 | "REAL CASH" UI rename (button + tooltip + confirm dialog) | 2dd612b |
| 5 | `PreTradeGate` — ws_connected, warmup 14 bars, SAFE_MODE, NaN/Inf guard | 80b3973 |
| 6 | `PositionSizingGate` — 0.5% per-trade, 2% daily, max 6 positions | 80b3973 |
| 7 | `_check_pre_trade()` wired at all 6 new-position sites in `main.py` | 80b3973 |
| 8 | `KillSwitch` slippage trigger (`slippage_pct_threshold=0.005`) | 951b900 |
| 9 | `drift_psi.py` per-category PSI thresholds + `MIN_PSI_SAMPLES=500` | 2a51631 |
| 10 | `ohlcv_parquet_loader.py` — FileNotFoundError, CSV.gz fallback removed | 6882836 |
| 11 | `pre_trade_gate.py` + `position_sizing.py` synced to VPS | manual SCP |
| 12 | rclone daily GDrive sync cron on VPS (3 AM UTC, excludes parquet) | VPS manual |
| 13 | Phase 1-4 VPS prep scripts (env_manifest, oos_signals, dataset_fingerprint) | various |
| 14 | Test suite: 2509 passed, 50 pre-existing failures, 0 regressions | 49cb07f |
| 15 | Parquet upload: 13,951 / 13,951 files, 23.75 GB, 0 errors | logs/parquet_upload_fast.log |

---

## FAIL-CLOSED DEFAULTS TABLE
*What blocks a new trade when a subsystem state is unknown*

| Subsystem | Unknown / error state | Action |
|---|---|---|
| `DASHBOARD_API_KEY` | blank / unset | Process refuses to start (SystemExit) |
| `MODEL_MANIFEST_KEY` | blank / unset | Process refuses to start (SystemExit) |
| WebSocket | not connected | PreTradeGate blocks new-position orders |
| Warmup | < 14 bars | PreTradeGate blocks |
| SAFE_MODE | True | PreTradeGate blocks new-position orders |
| Kill-switch | paused | `_check_pre_trade()` blocks all new orders |
| Model signature | missing / mismatch | model_integrity raises, no load |
| Drift | enforce-tier PSI >= 0.25 | `is_drift_paused()` → True immediately |
| Drift | confirm-tier PSI >= 0.25 for 3 consecutive hours | `is_drift_paused()` → True |
| NaN / Inf in features | any | PreTradeGate blocks |
| Funding blackout | 07:58–08:00 / 15:58–16:00 / 23:58–00:00 UTC | Futures orders blocked (C7) |

---

## PRODUCTION MODE MATRIX

| Mode | New spot | New futures | New scalping | Description |
|---|---|---|---|---|
| `PAPER` | mock only | mock only | mock only | All orders simulated |
| `TESTNET` | testnet | testnet | testnet | Real exchange, fake money |
| `REAL_CASH` | real | real | canary only | Operator explicitly enabled |
| `SAFE_MODE` | blocked | blocked | blocked | Triggered by PreTradeGate |

Transition rules: `PAPER → TESTNET → REAL_CASH` each requires explicit operator ГО. `SAFE_MODE` overrides any mode for new-position orders; close/reduce orders bypass.

---

## GLOBAL RULE: Config Schema Validation (MANDATORY)
*Applies to ALL hot-reload risk config files — never silently fall back to uncapped / default behavior on malformed JSON.*

Every file in `data/risk_*.json`, `data/capital_allocation*.json`, `data/strategy_config.json`:
1. Loaded via `safe_json.read_json()` (atomic read)
2. Validated with pydantic model OR explicit type/range checks immediately after load
3. On validation failure: **raise** (`ConfigValidationError`) — never silently apply defaults or uncapped values
4. Log the bad value and the rejection reason at ERROR level before raising
5. Tests: each config loader has a `tests/test_config_*.py` asserting that malformed JSON raises, not silently falls back

---

## REAL CASH READINESS CHECKLIST
*Must pass before any `REAL_CASH` mode GO. Script: `scripts/readiness_check.py`*

- [ ] All GATE 0 tests pass (0 failures)
- [ ] All GATE 1 model integrity tests pass
- [ ] All GATE 2 kill-switch E2E test passes (N8)
- [ ] `DASHBOARD_API_KEY` set and non-empty in VPS `.env`
- [ ] `MODEL_MANIFEST_KEY` set and non-empty in VPS `.env`
- [ ] All model `.sig` files present on VPS (Gate 1 / Z2)
- [ ] Dashboard auth working: unauthenticated request → 401
- [ ] Kill-switch `is_paused()` returns False on clean startup
- [ ] PreTradeGate passes warmup on clean startup
- [ ] WS connected flag True after 60s
- [ ] `drift_psi.py` baseline loaded (`data/risk/drift_baseline.json` exists)
- [ ] Funding blackout windows active (C7)
- [ ] Exchange precision cache loaded (C9)
- [ ] `data/audit/critical_alerts.jsonl` writable
- [ ] Rollback procedure documented and tested (N2)
- [ ] Paper trading 7-day validation passed for each active strategy (E4 canary)

---

## === GATE 0-A: Unicode / Encoding Safety ===  **P0**

### Z1 · Fix non-ASCII violations in Python source files ❌ BLOCKING
**python-reviewer FAIL.** `scripts/_unicode_audit_fix.py --dry-run` reports 82 replacements across 19 files.
Critical files with logger/print/raise violations (cp1252 unsafe on Windows):
- `src/main.py` — 36 violations (emojis + em-dashes in logger calls: lines 227, 295, 997, 1016, 1072, 1164, 1173, 1208, 1212, 1214, 1234, 1236, 1449, 1454, 1465, 1565, 1595/1598/1602, 1621, 1634, 1681)
- `src/utils/threshold_optimizer.py:80,111` — em-dashes in logger.warning
- `scripts/preflight_train.py:201,204` — em-dashes in print()
- `src/data_ingestion/coinglass_downloader.py:216,219,493,526,555` — 5 em-dashes
- `setup_telegram_auth.py:42,44`, `scripts/_wizard_fix_phase_a.py:124`, `src/data_ingestion/liquidation_downloader.py:87`
- `src/risk/live_perf_monitor.py` — check before committing (untracked file)

**Fix:** run `python scripts/_unicode_audit_fix.py` (no `--dry-run`), then manually verify `src/main.py`.
**Test (must add):** `tests/test_unicode_audit.py` — run `_unicode_audit_fix.py --dry-run` on repo, assert 0 replacements needed.
**Accept:** `python scripts/_unicode_audit_fix.py --dry-run` reports 0 violations.

---

## === GATE 0-B: Auth + Surface Hardening ===  **P0**

### A3 · VPS env startup guard ❌ BLOCKING
`grep -E "BINANCE|BINGX|ZMQ_BUS_KEY|DASHBOARD_API_KEY|MODEL_MANIFEST_KEY|HETZNER|VASTAI" /root/trading-bot/.env | cut -d= -f1`
Add startup guards at process start in `app.py` (not just a warning — `SystemExit`):
```python
if not os.environ.get("DASHBOARD_API_KEY", "").strip():
    raise SystemExit("FATAL: DASHBOARD_API_KEY not set")
if not os.environ.get("MODEL_MANIFEST_KEY", "").strip():
    raise SystemExit("FATAL: MODEL_MANIFEST_KEY not set")
```
**Accept:** blank key → process refuses to start with FATAL message in log.

### I1 · Authenticate all dashboard routes ❌ BLOCKING
**All 50+ routes in `src/dashboard/app.py`** must require `X-API-Key` header.
Special attention: `/api/scheduler/*`, `/api/cluster/worker_restart`, `/api/cluster/register` (3 NEW criticals).
- `app.py:206-207` — fix fail-open when `DASHBOARD_API_KEY` blank: startup guard from A3 handles this
- `/api/control` at `app.py:308` — `existing.update(data)` with no key allowlist: add explicit allowlist of mutable fields before merge
- Flask debug mode: enforce `debug=False` in production; `FLASK_ENV=production`
**Accept:** `curl -s http://127.0.0.1:5000/api/state` (no key) → 401 JSON response.

### I2 · Parameterized SQL / DuckDB queries ❌ BLOCKING
Audit all DuckDB injection vectors:
- String concatenation with `+` or `%` or `.format()`: `con\.execute\(.*(%|\.format\(|\+`
- Jinja2 SSTI: grep for `render_template_string` and `| safe` in templates
- Log injection: sanitize `str(e)` in all JSON responses (`str(e).replace('\n', ' ')`)
- `app.py:1845-1847`: `metric` string passed to `run_bake_off(metric=...)` without sanitization → validate against allowlist; `int(request.args.get(...))` raises ValueError → 500 with traceback: wrap in try/except returning 400
Replace user-controlled values with parameterized: `con.execute("SELECT ... WHERE col = ?", [value])`.
**Note:** 7 existing f-string PRAGMA/SET calls use hardcoded `pathlib.Path` constants — NOT user-controlled, not exploitable. Only fix calls that use request-derived values.

### I5 · Bind services to 127.0.0.1 ❌ BLOCKING
All Flask `app.run(host=...)` and cluster/monitor server bindings: use `127.0.0.1`, not `0.0.0.0`.
Exception: VPS dashboard on port 5000 where UFW already restricts access.
**Files:** `src/dashboard/app.py`, `src/engine/cluster_orchestrator.py`, `src/monitor/monitor_server.py`

### I6 · Delete / isolate `control_plane.py` ❌ BLOCKING
`src/dashboard/control_plane.py` (if it exists) — verify not reachable externally.
If unused: delete. If used: require auth + bind to 127.0.0.1 + confirm port is NOT :8100 (conflict with existing port).
**Accept:** no unauthenticated external-facing control surface.

---

## === GATE 1: Model Integrity — Sign + Verify ===  **P0**

### Z2 · Sign existing model artifacts ❌ BLOCKING (must precede I4)
Before adding HMAC-verify to `joblib.load()`, all existing `models/*.joblib` must be signed.
`scripts/sign_model_artifacts.py` — one-shot: reads each `.joblib`, computes HMAC-SHA256 with `MODEL_MANIFEST_KEY`, writes `.sig` file alongside.
Run once on both local AND VPS before I4 ships.
**Accept:** every `models/*.joblib` has a corresponding `.sig` file; script exits 0.

### I4 · Secure model loading ❌ BLOCKING (depends on Z2)
All `torch.load()` calls: add `weights_only=True`.
Trainers WRITE HMAC signature at save time — add `sign_and_save(artifact, path)` to all trainer save paths.
`model_integrity.py:307` — promote fail-open to hard error: check `MODEL_MANIFEST_KEY` at startup, same as A3.
TOCTOU: `verify_and_load_bytes()` open → verify → pass BytesIO to joblib — confirmed correct by security-reviewer.
**Files:** all trainers (save path) + `ml_predictor.py` + `multi_tf_predictor.py` + `regime_classifier.py` + `model_integrity.py`
**Tests:** `tests/test_model_integrity.py` — (a) load with missing `.sig` → raises; (b) load with bad signature → raises; (c) load with correct sig → succeeds; (d) `MODEL_MANIFEST_KEY` unset at import → `SystemExit`.
**Accept:** `MODEL_MANIFEST_KEY` unset → process refuses to start.

---

## === GATE 2: Live Risk Stop ===  **P0**

### C1 · Kill-switch slippage config field ❌
`slippage_pct_threshold` trigger exists in code (commit 951b900). Verify it is also in `KillSwitchConfig` dataclass as a configurable field (not hardcoded). If hardcoded: promote to named constant + `KillSwitchConfig` field.

### C2 · Full KillSwitch evaluator wiring ❌
Add to `src/risk/kill_switch.py`:
- `evaluate(ts)` polling: `daily_loss_R_multiple`, `consecutive_losses`, `latency_p99_ms` (feeds from N1), `drawdown_pct`, `calibration_brier_z`
- `pause()` / `reset()` / `state()` methods
- Dashboard: `GET /api/risk/kill_switch/status`, `POST /api/risk/kill_switch/reset`
- Dashboard tile in Audit tab: status, last trigger, last reset, [Reset] button
**N1 integration (arch):** `latency_p99_ms` data comes from `src/ops/latency_tracker.py` (N1). C2 and N1 are co-dependent — implement together.
**Accept:** synthetic test — 3R losses → `is_paused() == True`

### N5 · Cache `LOSSES_FILE` in KillSwitch ❌ (P0 — ships with C2)
`C2 KillSwitch._read_losses_count()` does disk I/O on every trade tick while holding `_lock`.
Cache `LOSSES_FILE` in-memory with short TTL (e.g. 5s). Background reader thread updates the cache off the hot path.
**Ships in the same PR as C2.**

### Z3 · Alerting channel — critical_alerts.jsonl ❌ BLOCKING
All "Telegram CRITICAL alert" references in code and plan replaced with:
1. Write to `data/audit/critical_alerts.jsonl` (NDJSON, append-only, atomic per-line write)
2. Surface as dashboard banner CRITICAL (poll endpoint: `GET /api/audit/critical_alerts/recent`)
3. Log to `logs/critical.log`
**No external channel.** No Telegram outbound. (Global rule: see memory.)
**Files:** `src/risk/kill_switch.py`, `src/ops/state_backup.py`, infra billing exception handlers

### N8 · E2E acceptance test: kill-switch → pre_trade → dashboard ❌ (P0 — must pass before E3 GO)
Integration test (NOT unit test — must exercise the real call chain):
1. Inject 3R consecutive losses into `KillSwitch` → assert `is_paused() == True`
2. Call `_check_pre_trade()` in `main.py` → assert it returns `(False, "kill_switch_paused")` for new-position order
3. Hit `GET /api/risk/kill_switch/status` → assert `paused == true` in JSON
4. Hit `POST /api/risk/kill_switch/reset` with API key → assert `is_paused() == False`
**Accept:** all 4 assertions pass. Ship before E3 GO.

---

## === GATE 3: Futures / Risk Math ===  **P1** (before E3)

### C7 · Funding-rate blackout windows ❌
Block new futures at 07:58–08:00, 15:58–16:00, 23:58–00:00 UTC.
Use `datetime.now(timezone.utc)` (NOT `datetime.utcnow()`).
**Implement C7 before I7** — both touch the futures order path; C7 first avoids merge conflicts.
**File:** `src/main.py` (futures order placement path)

### I7 · Risk math: liquidation + stop-distance + live funding ❌ (depends on C7)
Full type-annotated specs:
```python
def calc_liquidation_price(
    entry: float, leverage: float,
    margin_type: Literal['cross', 'isolated'],
    taker_fee: float, accumulated_funding: float
) -> float: ...

def size_from_stop_distance(
    equity: float, risk_pct: float,
    stop_dist: float, symbol_price: float
) -> float: ...
```
`live_funding.py` (new): shared ccxt instance, `threading.Lock`-protected TTL cache (60s), fail-closed on fetch failure.

### C9 · Exchange precision normalization ❌
Before every order: `exchange.amount_to_precision(symbol, raw_qty)` or Decimal rounding.
Fetch `step_size` / `tick_size` from `GET /api/v3/exchangeInfo`, 24h TTL cache.

### N6 · Centralize ExchangePrecisionCache ❌ (ships with C9)
`src/exchange/precision_cache.py` — single module, not inline per order path.
Otherwise duplicated across spot/futures/scalping callers.
**Ships in the same PR as C9.**

### C3 · Full PositionCaps ❌
`src/risk/position_caps.py` (new):
- Per-symbol cap 10% equity; total open exposure cap 50% equity; `HARD_CEILING` 5% per trade (immutable)
- Rejection feeds kill-switch `risk_cap_breach` event
Config: `data/risk_caps.json` (hot-reload, **schema-validated per global config rule above**)
**Invariant assertion at startup:** `assert PositionSizingGate.per_trade_pct <= PositionCaps.HARD_CEILING`
**Arch note (K3 interaction):** K3 Master Allocator sets `capital_limit` as TARGET weight; C3/C6 caps clip post-hoc. Allocator → PositionCaps → LeverageCap is the evaluation order in `_check_pre_trade()`.
**Tests:** `tests/test_risk_position_caps.py` — per-symbol cap, total cap, hard ceiling enforcement, malformed JSON hot-reload rejection.

### C6 · Leverage cap enforcer ❌
`src/risk/leverage_cap.py` (new): max total 3x, per-strategy 2x, per-symbol 1.5x.
`deleverage_actions()` closes worst performer first. Kill-switch polls every tick.

---

## === P1 BLOCK: Required Before E3 Retrain ===

### N1 · Latency tracker ❌ (P1 — feeds C2 kill-switch)
`src/ops/latency_tracker.py` — record per-order RTT (signal emission → exchange ACK timestamp).
Expose `p99_ms()` method. Kill-switch C2 reads `latency_p99_ms` from this module.
**Integrate with C4 tick circuit breaker (C4 also uses per-symbol timing data).**
**Tests:** `tests/test_latency_tracker.py`

### N2 · Model rollback procedure ❌ (P1 — before E3)
E4 Champion/Challenger handles new-vs-challenger. But also need:
`rollback_to_previous(model_key)` in `baseline_manager.py` — triggers on consecutive live losses (not just training KPI fail).
**Rollback drill:** cron or manual script that (a) snapshots current model, (b) swaps to previous, (c) validates bot reads new path, (d) documents rollback in `data/audit/rollback_log.jsonl`.
Drill must be run and documented BEFORE E3 GO.

### A2 · One-time GDrive parquet backup ❌ (P1)
On VPS after A1 (DONE): `rclone copy /root/trading-bot/data/parquet/ gdrive:trading-bot-backup/parquet-archive/`
**Accept:** `rclone lsd gdrive:trading-bot-backup/parquet-archive/` returns entries.

### A4 · Reset agent heartbeats on VPS ❌ (P1)
Write fresh `data/agent_status.json` — all `status: inactive`, `last_heartbeat: null`.

### A5a · CSV.gz → Parquet migration — write phase ❌ (P1)
Write all batches to `data/parquet_us/` (original `data/parquet/` untouched).
Validate: file count + schema + fingerprint.
Force `datetime64[us]` timestamps (PyArrow 13+, do NOT use `ns`).
Add back-compat reader-layer cast for any unmigrated tool that reads parquet directly.

### A5b · Cutover — stop bot → swap → restart 🚫 (P1 — separate GO, pre-checklist required)
**Pre-checklist (must verify before issuing cutover GO):**
- [ ] A5a write phase complete: `parquet_us/` file count matches original `parquet/` count
- [ ] Schema validated: no timestamp dtype regressions
- [ ] Fingerprint hash recorded in `data/audit/migration_fingerprint.json`
- [ ] GDrive backup (A2) verified fresh
- [ ] Bot has no active open positions
- [ ] Maintenance window chosen (low-activity UTC hours)
- [ ] Rollback procedure documented: `mv data/parquet/ data/parquet_us/` → `mv data/parquet_backup/ data/parquet/`
- [ ] Post-restart checks: bot reads parquet, training works, no schema errors in logs for 10 min

Cutover sequence: confirm no running jobs → stop bot → `mv data/parquet/ data/parquet_backup/` → `mv data/parquet_us/ data/parquet/` → start bot.
Delete `data/parquet_backup/` only after first successful training confirms correctness.
Move CSV.gz → `data/raw_archive/`. Archive cleanup cron: `0 4 * * * find .../raw_archive/ -name "*.csv.gz" -mtime +7 -delete`
> Architect: split from monolithic A5 to avoid stopping bot during dashboard testing.

### I3 · ZMQ HMAC signing + replay protection ❌ (P1)
All ZMQ pub/sub channels must sign messages with `ZMQ_BUS_KEY`.
Publisher: include monotonic sequence number OR UTC timestamp (`±5s` window) INSIDE `body` before msgpack-packing. Then append HMAC-SHA256 of the full body as last frame.
Subscriber: verify HMAC; check timestamp within ±5s OR maintain per-connection ring buffer (deque, last 1000 nonces) for replay detection.
**Security finding (HIGH):** HMAC alone without nonce/timestamp allows indefinite replay from local socket access.
Add key rotation procedure to runbook (e.g. rotate `ZMQ_BUS_KEY` via `.env` update + coordinated restart).
**Files:** all files importing `zmq`

### B10 · Trade enrichment fields — going-forward ❌ (P1 — gates J5, C5)
Add to every trade write: `mode`, `regime_at_entry`, `model_confidence`, `mfe_pct`, `mae_pct`, `slippage_pct`, `exit_reason`
**Files:** `src/engine/trade_tracker.py`, `src/engine/paper_book.py`

### J1 · Two-tier drift enforcement (enforce vs confirm) ❌ (P1)
**Status:** Phase 11 added PSI thresholds. The enforce/confirm tier LOGIC in `drift_monitor.py` is still unimplemented.
- `src/risk/drift_psi.py`: add `_enforce_features()` reading `DRIFT_ENFORCE_FEATURES` env var; tag `DriftFinding.is_enforce_feature: bool`
- `src/risk/drift_monitor.py`: `CellState` add `consecutive_pause_count: int = 0`; `is_drift_paused()` returns True on (a) any enforce-tier finding OR (b) `consecutive_pause_count >= 3`
- `.env.example`: document `DRIFT_ENFORCE_FEATURES=ofi_z,funding_z,macd_hist,frac_diff_d40`
**Tests:** enforce-tier at PSI>=0.25 halts immediately; confirm-tier halts only at count=3

### J2 · Overfitting ratio in trainers + kpi_gate ❌ (P1 — must land before E3)
All 5 trainers:
- WF fold loop: collect `in_sample_fold_accs` alongside `fold_accuracies`
- After loop: `in_sample_mean = mean(in_sample_fold_accs)`; guard: `overfit_ratio = (in_sample_mean - wf_mean) / in_sample_mean if in_sample_mean > 0 else float('inf')`
- Log WARNING if >0.10; ERROR if >0.20
- Save `in_sample_mean_acc`, `overfit_ratio` to meta.json
`kpi_gate.py`: `TrainingResult.overfit_ratio: float | None = None`, max-check in `_check_thresholds()`, read in `evaluate_from_meta_json()`
`data/training_rules.json`: add `"overfit_ratio": 0.20` per model

### J3 · Per-fold WF scores saved from trainers ❌ (P1 — must land before E3)
> python-reviewer: slope gate is ALREADY implemented in `kpi_gate.py`. Only missing piece: trainers don't save `wf_fold_scores` to meta.json yet.
All 5 trainers: collect existing per-fold `fold_accuracies` list → save `"wf_fold_scores": fold_accuracies` (time-ordered) to meta.json.
`kpi_gate.py` changes: `TrainingResult.wf_fold_scores: list[float] | None = None`, `evaluate_from_meta_json()` reads the field.

### J4 · Per-strategy regression guard in auto_retrain ❌ (P1)
`src/engine/auto_retrain.py`:
- `_per_strategy_regressions(before, after, tolerance) -> list[str]`
- Call after computing before/after; if ANY strategy regresses individually → verdict = "regression"
- Add `per_strategy_before/after/delta`, `regressed_strategies`, `new_strategies` to output dict
**Tests:** one strategy drops 10%, another improves 20% → verdict = "regression"

### J5 · Wire `live_perf_monitor.py` into main.py + dashboard ❌ (P1 — depends on B10)
> Planner: `record_outcome()` needs `exit_reason` + `slippage_pct` from B10 trade enrichment fields.
`src/risk/live_perf_monitor.py` exists (untracked). Steps:
1. Commit the file (after Z1 non-ASCII fix — check for violations first)
2. `record_signal()` / `record_outcome()` must be non-blocking: use `queue.Queue`; hourly `_loop` thread drains queue during `run_once()`. **Do NOT do file I/O on the hot bot-loop tick.**
3. Wire `record_signal()` after ML signal emission in `src/main.py`
4. Wire `record_outcome()` on trade close in `src/main.py`
5. Fix `live_perf_monitor.py:179`: replace `datetime.fromtimestamp(datetime.now(tz.utc).timestamp() + N, tz=...)` with `datetime.now(tz.utc) + timedelta(seconds=N)`
6. Dashboard: `GET /api/risk/live_perf/state`, tile on Risk tab
7. J5 feeds: `auto_retrain.py` (DEGRADED → trigger retrain), kill-switch evaluator (DEGRADED → pause), J4 regression guard

### I8 · Behavioral test rebuild ❌ (P1)
Audit `tests/test_dashboard.py` for string-match-only assertions (`'def my_function' in source_text`).
Convert each to a behavioral test: call the function, assert on return value or side effect.
**Accept:** 0 string-match-only test paths; all tests call code under test.

### N3 · Requirements freeze before E3 ❌ (P1)
`scripts/freeze_requirements.py` → `data/env_snapshots/train_env_YYYY-MM-DD.txt`
Train vs serve environment mismatch breaks reproducibility. Uses `env_manifest.py` (already exists).
Run and commit before issuing E3 GO.

### N7 · Module tests — added per module as it ships ❌ (P1 — ongoing)
> Operator restructure: tests added per module, NOT in one big batch at the end.
Each new module ships with its test file in the same PR:
- C3 → `tests/test_risk_position_caps.py`
- C4 → `tests/test_tick_circuit_breaker.py`
- C5 → `tests/test_risk_correlation.py`
- C6 → `tests/test_risk_leverage_cap.py` (ships with C6)
- I7 → `tests/test_live_funding.py`
- G1 → `tests/test_state_backup.py`
- G3 → `tests/test_state_reconciler.py`
- G4 → `tests/test_network_health.py`
- G5 → `tests/test_capital_allocator.py`
- G7 → `tests/test_exchange_health.py`
- K2 → `tests/test_risk_strategy_scorer.py`
- K3 → `tests/test_risk_master_allocator.py`
- K4 → `tests/test_risk_decay_monitor.py`
**Rule:** no module merges without its test file. 0 failures gate on every push.

---

## === P2: After Baseline Production Hardening ===

*The items below wait until the system is fail-closed, authenticated, signed, observable, and capable of stopping trading on its own.*

---

## PHASE A (remaining) — VPS Data Setup  **P2** (except A2/A4/A5 above which are P1)

### A1 · Parquet upload ✅ DONE
13,951 / 13,951 files, 23.75 GB, 0 errors (logs/parquet_upload_fast.log).

---

## PHASE B — Dashboard & Training Fixes  **P2**

### B1 · Curated `DEFAULT_PER_KEY_TFS` map ❌
`src/engine/train_all_models.py`
```python
DEFAULT_PER_KEY_TFS = {
    'base':     ('5m', '15m', '1h', '4h', '1d'),
    'trend':    ('15m', '1h', '4h', '1d', '1w'),
    'futures':  ('5m', '15m', '1h', '4h', '1d'),
    'scalping': ('1m', '5m'),
    'meta':     ('5m', '15m', '1h', '4h'),
    'tft':      ('15m', '1h', '4h'),
    'regime':   ('1h',),
}
```

### B2 · Per-model + per-TF `run_full_backtest` filter ❌
Add `models: tuple | None` and `timeframes: tuple | None` params.
**Files:** `src/engine/backtester.py`, `src/dashboard/app.py`

### B3 · TFT `cannot reindex` deeper fix ❌
Dedupe before EACH of the 3 `from_dataframe()` calls in `build_series_bundle`.
If `asfreq` is source: set `freq` from actual median bar interval.
**File:** `src/engine/train_tft_model.py`

### B4 · TFT regression test ❌
Synthetic small dataframe with duplicate timestamps + irregular gaps.
`build_series_bundle(df, freq='1h')` — assert no `ValueError`, consistent indices.
**File:** `tests/test_dashboard.py` (Phase 70) | **Deps:** B3

### B5 · Scalping label rebalance ❌
`class_weight='balanced'` to HistGBT in `src/engine/train_scalping_model.py`.
Layer in `imblearn.SMOTE` on training fold only if still imbalanced.
**Accept:** `long_acc >= 50%` AND `short_acc >= 50%`

### B6 · OFT sweep coverage ❌
Add OFT to `train_all_models.train_all()` — single canonical TF `1m`, per-symbol loop.
Verify `_RESOURCE_KIND['oft'] = 'exclusive'`.

### B7 · 1s archive coverage check ❌
Per symbol: `gunzip -c data/raw/historical/<sym>_USDT_spot_1s.csv.gz | tail -1`
Check 6 stale symbols: BNB / DOGE / ETH / LINK / TRX / XRP
Output: `data/audit_reports/data_coverage_check_2026_05_08.md`

### B8 · Mode-aware Portfolio loader ❌
`loadPortfolioByMode()` in dashboard JS — fetches `/api/portfolio?mode=<m>`.
PAPER → USDT-only rows; TESTNET → from exchange.
**Files:** `src/dashboard/templates/index.html`, tests Phase 72

### B9 · Per-market Signal & Risk panels ❌
Replace single-market panels with 3 stacked rows (SPOT / FUTURES / SCALPING).
Signal: `active_symbol / last_signal / sentiment / rsi`. Risk: `vol / size / open_count`.

### B11 · Backfill 912 historical trades ❌
`scripts/backfill_trade_enrichment.py` → `data/trades_enriched.json`
**Deps:** B10 (P1)

### B12 · 1s archive refill (only if B7 finds gap) ❌
`python -m src.data_ingestion.binance_archive_downloader --symbols <list> --start 2024-12-31`

### B13 · "Backfill missing data" button ❌
Endpoint `/api/data/backfill` + UI button + progress chip.

### B14 · Cold-start disk cache ❌
`src/dashboard/cold_cache.py` — saves/loads status caches to `data/cache/cold/`.
**Accept:** cold-start `/api/db/status` < 100 ms

---

## PHASE C (remaining) — Risk Controls  **P2** (C1/C2/C3/C6/C7/C9 are P0/P1 above)

### C4 · Tick circuit breaker ❌
`src/risk/tick_circuit_breaker.py` (new):
4-sigma spike → freeze symbol for 60s. Rolling 30-bar deque, per-symbol.
Log: `data/audit/circuit_breaker_log.jsonl`
**N1 integration:** use latency_tracker.py timestamps. Ships with `tests/test_tick_circuit_breaker.py`.

### C5 · Correlation gate + monitor ❌ (depends on B10 + B11)
- `src/risk/correlation_gate.py` — order-time: Pearson >0.7 (30-day rolling) → max 20% in cluster. Needs `src/risk/return_series_cache.py` (new): builds rolling 30-day per-strategy return series from `data/trades_enriched.json`. Cache refreshed hourly.
- `src/risk/correlation_monitor.py` — daily P&L Spearman matrix; alarm >0.7; dashboard heatmap
**Dep:** B10 (P1) + B11 must land first.
Ships with `tests/test_risk_correlation.py`.

### C8 · Clock drift monitor ❌
Startup + every 5 min: `|local_ms - exchange_server_ms| > 500ms` → alert.
Fix: `chronyc makestep` (Ubuntu 24.04).

### C10 · Minimum liquidity filter ❌
At order-generation time (60s TTL cache):
- 24h volume >= $50M spot / $100M futures
- Bid-ask spread <= 0.05% spot / 0.03% futures
- Book depth >= $50K at 0.1% from mid

### C11 · State reconciliation on WS reconnect ❌ (depends on G3)
> Architect: G3 is the canonical reconciler. C11 calls into G3.
On every reconnect BEFORE setting `ws_connected = True`: call `state_reconciler.reconcile()` (G3 module).
Mismatch → SAFE_MODE = read_only, surface in dashboard.
**Order idempotency:** add `clientOrderId` (UUID) to every order placement. On reconnect, check for UUID already submitted before placing — prevents double-fill.

---

## PHASE D — Validation Pipeline  **P2**

### D1 · Sprint 1a R1 — Per-model agent refactor ❌
Split monolithic `train_all_models.py` into per-model files.
Orchestrator becomes topic dispatcher. Delete orphan detection from dashboard.
**Effort:** 5-7 days

### D2 · Sprint 1a R2 — KPI gate per run ❌
`data/training_rules.json` gains `kpi_threshold` block: `walk_forward_sharpe`, `calmar`, `max_drawdown_pct`, `win_rate`, `expectancy`, `min_total_trades`.
3 consecutive misses → auto-retire via `strategy_registry.py` status flag.

### D3 · Sprint 1a R3 — Model Comparison dashboard tab ❌
Sortable KPI grid (model x symbol x TF rows) + drill-down + Promote/Retire/Restore buttons.

### D4 · Audit scaffolding ❌
`data/audit/` subdirs + `src/validation/__init__.py` + `src/audit/__init__.py` (stubs)
`src/validation/types.py` (`ValidationReport` dataclass)
`strategy_registry.py`: add `validation_status: Literal['live','shadow','killed','unaudited']`
Dashboard Audit tab placeholder (6 empty sub-tiles)

### D5 · Vol-adjusted Triple Barrier labeling ❌
`src/validation/vol_adjusted_barriers.py`: `PT = k1*sigma_t`, `SL = k2*sigma_t`. Default k1=1.8, k2=1.2 (asymmetric PT>SL — consistent with AFML Chapter 3).
`vol_window` must be **parameterized per-TF**:
```python
VOL_WINDOW_BY_TF = {'1m': 200, '5m': 100, '15m': 60, '1h': 30, '4h': 20, '1d': 14}
```
Trainers gain `--label-scheme=vol_adjusted` flag (default: `fixed` for back-compat).
Gate: decide label scheme BEFORE E3 retrain.

### D6 · Walk-forward harness + embargo ❌
`src/validation/walk_forward_harness.py`: config 60/14/14 days, min_folds=4.
Embargo must be `timedelta` per-TF, NOT a flat fraction:
```python
EMBARGO_BY_TF = {'1m': timedelta(hours=4), '5m': timedelta(hours=6),
                 '15m': timedelta(hours=12), '1h': timedelta(days=2),
                 '4h': timedelta(days=4), '1d': timedelta(days=7)}
```

### D7 · Feature leakage detector ❌
`src/validation/leakage_detector.py`: future-bar correlation, rolling-window includes-current-bar (AST walk), normalization across full dataset. `severity='high'` → blocks live promotion.

### D8 · Adversarial validation ❌
`src/validation/adversarial_validator.py`: AUC > 0.65 on (train vs live) classifier → kill.

### D9 · `validate_model()` + wire into trainers ❌
Single entry point in `src/validation/__init__.py`. `verdict=='kill'` → RuntimeError (blocks joblib write).

### D10 · Forecast model bake-off ❌
`src/audit/forecast_bakeoff.py`: TFT vs LightGBM vs CatBoost vs XGBoost across 1s/5s/1m/5m/15m horizons.

### D11 · Path-optimizer bake-off ❌
`src/audit/path_optimizer_bakeoff.py`: OFT-RL vs Dijkstra vs Bellman-Ford vs A* vs round-robin.

### D12 · Cut list builder ❌
`src/audit/cut_list_builder.py`: Sharpe >1.0 → live; 0.5-1.0 → shadow; <0.5 → kill.

### D13 · Execute cut list ❌
For each kill: delete registry entry, archive joblib, remove trainer file, document.
**Gate:** D12 complete + operator review

---

## PHASE E — VPS Retrain  **P2**
**All sub-phases require their own GO. Requires ALL P0 + P1 complete.**

### E1 · Smoke-test synthetic data 🚫
CPU (Hetzner CCX33) + GPU (Vast.ai RTX 4090). >=50k CPU rows, >=200k GPU rows, exact OHLCV_COLS schema.
Server/instance DELETED/DESTROYED via API with 3-attempt exponential backoff after test.

### E2 · Archive training state 🚫 (blocked by E1)
Stop bot. Archive `models/` + training JSONs → `data/training_archive/YYYY-MM-DD/`. Clear originals.

### E3 · Retrain from scratch 🚫 (blocked by E2 + ALL P0 + P1 complete)
Training order: `regime → base → trend → futures → scalping → meta → tft → oft`
Rules:
1. Scalping paper/experimental (>=500 trades, >=30 days canary before live)
2. Optimize for after-fee Sharpe + profit_factor >1.5
3. Validate per regime (bull/bear/chop/high_vol)
4. First live: only Combo A/B/C (Trend RF + Meta + Regime)
5. Disable rule strategy zoo day 1 (ElliottWave, Ichimoku, MACD Div, etc.)
6. `execution_audit.jsonl` mandatory from day 1
7. Test combos in sequence (Stage 1-5, >=50 trades each before promoting)
8. TFT/OFT/scalping paper-only on first live deployment
DuckDB: `SET memory_limit='18GB'`, `SET temp_directory='/root/.../duckdb_temp/'`, singleton connection.

### E4 · Champion/Challenger baseline system 🚫 (blocked by E3)
- `src/governance/baseline_manager.py` — PromotionPolicy + rollback()
- `src/agents/model_comparison_agent.py` — supervised comparison loop
- `src/utils/trading_cost_stress_test.py` — fees×1.5, slippage×2, latency spikes
- Canary: Scalping >=500 trades/14d; Base/Meta >=100 trades/14d; Trend/Futures >=50 trades/30d

---

## PHASE F — Post-Retrain Analytics  **P2**
**Blocked by E3**

### F1 · Overnight retrain sweep — 26 combos + chained backtest ❌
### F2 · Post-retrain accuracy audit ❌
### F3 · Multi-TF cross-TF confirmation gate ❌
### F4 · 1-week paper-trading validation ❌

### F5 · Analytical dashboard — 7 sections ❌
1. Strategy P&L heatmap (strategy x symbol x TF, cell=Sharpe)
2. Regime-conditional performance
3. Training history (WF acc / AUC over time)
4. Calibration plots
5. Trade lifecycle distributions (MFE/MAE/exit histograms)
6. Slippage + fee cumulative drag
7. Correlation matrix (strategy returns)
Sources: ParquetClient + `data/trades_enriched.json`. Nightly aggregation script.

### F6 · FastAPI process separation ❌
Move `/api/db/status`, `/api/db/market_stats`, `/api/db/training_history` to FastAPI control plane on port **:8200** (NOT :8100 — that port conflicts with `control_plane.py` per I6).
Flask proxies. **Deps:** B14

---

## PHASE G — Operational Hardening  **P2**

### G1 · Automated state backups every 4h ❌
`src/ops/state_backup.py` — tar.gz of `data/` (excl parquet) + `models/` + configs.
Output: **`/root/backups/<YYYYMMDD_HHMMSS>/`** on VPS (Linux path). Retention: hourly×24 / daily×7 / weekly×4 / monthly×INF.
Scheduled restore drill: quarterly cron restores random recent backup to `/root/backups/restore_test/` and validates file count + schema.
Critical failure → `data/audit/critical_alerts.jsonl` + dashboard banner (see Z3).
Ships with `tests/test_state_backup.py`.

### G2 · Offsite encrypted snapshots (Backblaze B2) ❌
AES-256-GCM encrypt → B2.
**Security (HIGH):** Use HKDF key derivation: `per_backup_key = HKDF(master_key, info=backup_date_str)`. Store GCM nonce/IV alongside ciphertext. Document rotation procedure.
Daily incremental + weekly full.

### G3 · Restart reconciler ❌
`src/ops/state_reconciler.py` — on startup: pull live positions/orders, compare to local state.
Orphan orders, phantom positions → `RECONCILE_REQUIRED`, refuse to trade until operator ACKs.
Ships with `tests/test_state_reconciler.py`.

### G4 · Network outage SAFE MODE ❌
`src/ops/network_health.py` — heartbeat to each exchange every 30s.
3 consecutive fails → SAFE MODE. On recovery: run G3 first.
Critical alert → `data/audit/critical_alerts.jsonl` + dashboard banner (see Z3).
Ships with `tests/test_network_health.py`.

### G5 · Multi-exchange capital split ❌
`src/risk/capital_allocator.py` — `pick_exchange_for_order()`.
Config: `data/capital_allocation.json` (default: Binance 50% / Bybit 30% / OKX 20%).
Ships with `tests/test_capital_allocator.py`.

### G6 · Auto-withdraw to cold storage ❌
Weekly, operator-ACK required, `enabled: false` default.
Cold address whitelist: file edit only (not dashboard).

### G7 · Exchange health monitor ❌
`src/ops/exchange_health.py` — API latency p99, ack rate. 3 RED signals → quarantined.
Ships with `tests/test_exchange_health.py`.

---

## PHASE H — Drift Monitor Extensions  **P2**

### H1 · Two-level PSI (hourly rolling + daily KL) ❌
Hourly: PSI vs rolling 24h window (NOT static training dist).
Daily: KL divergence vs training distribution.
Reference distributions saved at training time: `data/baselines/vN/train_distributions/`
**Never automatic retrain** — operator decides.

### H2 · Feature z-score + volatility drift ❌
Feature drift: mean/std >3-sigma per-feature z-score.
Vol drift: >2-sigma rolling 30-day → regime change signal.

---

## PHASE K — Post-Production Orchestration  **P2**
*Starts AFTER E3 retrain + >=30 days live trading.*

### K1 · Strategy taxonomy enforcement ❌
Formalize 3-tier taxonomy in `strategy_registry.py`:
- **Core** (receive capital): Trend Momentum RF (1h/4h), Volatility Breakout (15m/1h), Mean Reversion (5m/15m chop-only)
- **Filters** (gate signals, no independent capital): Meta-labeler, Regime GMM, GARCH sizing, SMA-200 macro, Liquidity filter, Correlation gate
- **Experimental** (paper only): TFT, OFT, extra Base RF TFs
Add `tier: Literal['core', 'filter', 'experimental']` field.

### K2 · Strategy Score (composite, hourly) ❌
`src/risk/strategy_scorer.py` (new). Composite: rolling 7d live Sharpe (40%) + Sharpe deviation from backtest baseline (30%) + 5d EMA win rate trend (15%) + slippage cost trend (15%).
`DECAY_THRESHOLD: float = 0.5` (named constant). State: `data/risk/strategy_scores.json`.
Ships with `tests/test_risk_strategy_scorer.py`.

### K3 · Master Allocator ❌
`src/risk/master_allocator.py` (new). Allocator → PositionCaps → LeverageCap evaluation order.
- `active = [s for s in strategies if s.score > 30 and regime.allows(s)]`
- `weights = softmax([s.score for s in active])`; single-strategy weight cap (max 40%)
- `s.capital_limit = total_capital * w * (1 - correlation_penalty(s))`
- **Min-dwell:** allocation change requires >=24h since last change per strategy
Ships with `tests/test_risk_master_allocator.py`.

### K4 · Strategy Decay Monitor ❌
`src/risk/decay_monitor.py` (new):
- Rolling 3d Sharpe < 0.5 AND declining → "Degrading" (reduce allocation 50%)
- Rolling 3d Sharpe < 0.0 → "Suspended" (paper only)
- Recovery: 5d Sharpe > 1.0 → operator manual re-enable
State: `data/risk/decay_state.json`. Ships with `tests/test_risk_decay_monitor.py`.

### K5 · Capital allocation architecture by bankroll ❌
Config: `data/capital_allocation_config.json`. Auto-select profile by `account_balance`.
- **$1k profile:** BTC+ETH+SOL only, max 3-4 positions, 0.25-0.5% risk/trade, 1-2x leverage
- **$10k profile:** adds funding carry (spot+perp delta-neutral), multi-venue intelligence
Fallback: $1k profile until explicitly upgraded.

### K6 · Funding anomaly as meta/regime feature ❌
Add `funding_percentile`, `funding_z_score`, `funding_regime` to meta model + regime layer.
**Deps:** E3 retrain

### K7 · BTC lead-lag signal ❌
BTC futures impulse → alt lag → signal feature. `src/analysis/lead_lag_features.py` (new).
**Deps:** E3 retrain

### K8 · Statistical spread models ❌
ETH/BTC spread mean reversion + SOL/ETH relative momentum. `src/analysis/spread_features.py` (new).
**Deps:** E3 retrain

---

## PHASE J (remaining, P2 only)

### J1-J5 are P1 (listed above in P1 block)

---

## N-items Summary

| ID | Item | Priority | Ships with |
|---|---|---|---|
| N1 | Latency tracker → feeds C2 kill-switch | P1 | C2 |
| N2 | Model rollback procedure + drill | P1 | before E3 |
| N3 | Requirements freeze | P1 | before E3 |
| N4 | Fee/funding reconciliation vs exchange statements | P2 | G* block |
| N5 | Cache LOSSES_FILE in KillSwitch | P0 | C2 |
| N6 | Centralize ExchangePrecisionCache | P1 | C9 |
| N7 | Per-module tests (ongoing, ship with each module) | P1 | per PR |
| N8 | E2E kill-switch acceptance test | P0 | Gate 2 |

---

## Infrastructure Billing Rules (MANDATORY)
- **Hetzner:** DELETE server (never power off). Delete in exception handler with exponential backoff (30s → 60s → 120s, 3 attempts). Confirm via API. Log server ID. Failure → `data/audit/critical_alerts.jsonl` + dashboard banner CRITICAL.
- **Vast.ai:** DESTROY instance (never stop). Destroy in exception handler with exponential backoff. Confirm via API.

---

## Conflict Resolutions (carried from prior plans)

| Conflict | Resolution |
|---|---|
| Position caps 0.5% (VPS) vs 1% (TECH) | Keep `PositionSizingGate` at 0.5%. C3 adds full `PositionCaps` with `HARD_CEILING` 5%. Operator configures via `risk_caps.json`. |
| Kill-switch pre-trade (done) vs full evaluator (C2) | Pre-trade gate (80b3973) handles ws/warmup/SAFE_MODE. C2 adds drawdown%/losses/latency/Brier. Complementary. |
| Correlation order-gate vs P&L monitor | Both needed. C5 `correlation_gate.py` = enforcement; `correlation_monitor.py` = daily alerting. Same 0.7 threshold. |
| Execution audit (VPS) vs TECH S0-4 | Same `execution_audit.jsonl`. VPS schema (meta_passed, regime_used, garch_used) is canonical. TECH S0-4 is the API layer on top. |
| Analytical dashboard (PLAN_2026) vs Audit tab (TECH) | TECH Audit tab = validation/bakeoff/kill-switch (Phase D). Analytics tab = P&L matrix/regime perf/trade lifecycle (Phase F5). Different tabs. |
| F6 port :8100 (old) vs control_plane.py | Corrected to :8200. I6 isolates/deletes control_plane on :8100. |
| "Telegram CRITICAL" in billing rules | Replaced with `data/audit/critical_alerts.jsonl` + dashboard banner (Z3). No Telegram outbound. |
