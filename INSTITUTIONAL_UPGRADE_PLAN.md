# Institutional-Grade Bot Upgrade — Master Plan

Approved 2026-05-01. Source: `updated_architecture_plan_en.md`.

This is the durable, AI-readable implementation plan. Any agent (Claude, Cursor, Copilot, etc.) picking up the work should read this file first.

## Progress (auto-updated)

| Phase | Status | Tests |
|---|---|---|
| 0 — Foundation (DB + transport) | ✅ DONE | 55 PASS |
| 1 — L1 Data Layer | ✅ DONE | 53 PASS |
| 2 — L2 Alpha Engine | ✅ DONE | 35 PASS |
| 3 — L3 Execution & Simulation | ✅ DONE | 36 PASS |
| 4 — L4 Portfolio Optimization | ✅ DONE | 23 PASS |
| 5 — L5 Institutional Safeguards | ✅ DONE | 26 PASS |
| 7 — Continuous pipeline + retention | ✅ DONE | 39 PASS |
| 8 — Data governance + rate limiting + 11 connectors | ✅ DONE | 40 PASS |
| 9 — Integration + analytics + dual-balance + 6 more connectors | ✅ DONE | 41 PASS |
| 6 — Dashboard 8-tab nav + REAL/TEST switcher | ✅ DONE | (covered by P10 tests) |
| 10 — Live wiring (Parquet-first, alpha-decay, dyn threshold, β-history, news, train_v2) | ✅ DONE | 38 PASS |
| Multi-timeframe migrations (1m/1h/1d/funding) | ✅ DONE | 26 PASS |
| 1-sec data migration (34 GB → 49.75 GB Parquet, 3.27 B rows) | ✅ DONE | — |
| 1m/1d/1mo Binance archive download | ✅ DONE | — |
| Documentation: `APP_DOCUMENTATION.md` | ✅ DONE | — |
| Dashboard regression suite | ✅ | **503 PASS** |

## Phase 7 — Continuous Pipeline & Retention (Added)

New files:
- `src/data_ingestion/realtime_db_writer.py` — Binance WS → QuestDB ILP, only emits closed bars (`k.x=true`); nightly QuestDB → Parquet rollover.
- `src/data_ingestion/startup_recovery.py` — on every system start: gap-fill via archive (full months) + REST top-up (recent <30d); resumes realtime cleanly.
- `src/database/retention_manager.py` — tracks (symbol, tf, yyyymm) partitions, marks them as trained-on by which models, identifies archive-eligible.
- `src/database/google_drive_backup.py` — pydrive2 wrapper (graceful no-op when creds absent); archives parquet partitions to a Google Drive folder.
- `src/data_ingestion/binance_archive_downloader.py` — generalized: now supports `--timeframe 1m/1h/1d/1mo` and `--all-timeframes`. 1s legacy paths preserved.

Modified:
- `restart_all.ps1` — runs `startup_recovery` after QuestDB is up; spawns `realtime_db_writer` as a managed process (PID tracked alongside bot/dashboard).


---

## 1. Approved Stack

| Layer | Now (Phase 0–6) | Migration Later | Trigger |
|---|---|---|---|
| Data store | DuckDB + Parquet | ClickHouse | Parquet store > 500 GB OR backtest > 5 min |
| Control plane | FastAPI | FastAPI (keep) | — |
| Data plane | ZeroMQ | Kafka | 3+ training nodes OR need replay/event sourcing |
| Training | PyTorch native | PyTorch native (keep) | — |
| RL | SAC primary + PPO backup | Same | — |
| Risk solver | CVXPY | CVXPY (keep) | — |
| Frontend | Flask + Chart.js (JSON-first) | React + WebSocket | Drag-zoom heatmaps, multi-pane resize |

**Design seams that make migrations 1-day swaps:**
- `src/database/parquet_store.py` exposes `query(symbol, start, end) → DataFrame` — same API for ClickHouse later
- `src/transport/data_bus.py` exposes `publish_orderflow / subscribe_orderflow / push_batch / pull_batch` — same API for Kafka later
- Dashboard routes return JSON; Jinja templates only consume them — React drop-in later

---

## 2. Database Strategy

**Hot path (real-time):** QuestDB Docker (existing, ports 9000/9009/8812). Stays as-is. Stores last ~30 days of streaming data within 4 GB cache.

**Cold/historical path:** DuckDB + partitioned Parquet at `data/parquet/{symbol}/{YYYY-MM}/data.parquet`. Stores all 1-second tick history (already downloaded as CSVs). DuckDB queries Parquet directly via SQL — no migration overhead per query.

**Migration script:** `scripts/migrate_1sec_to_parquet.py` — one-time CSV → Parquet conversion. Idempotent (skips already-converted months).

**Sizing estimate:** 10 symbols × 1 year of 1-sec data ≈ 315 M rows ≈ 15 GB compressed Parquet (~100 GB raw CSV).

---

## 3. Distributed Training Architecture

```
MAIN PC (this machine)                            SECOND PC
─────────────────                                 ─────────
QuestDB (Docker)                                  Worker node
DuckDB + Parquet store                            RTX training
FastAPI control plane    ◄── REST commands ──►    
ZeroMQ data bus          ─── PUSH batches ──►     PULL batches
                         ◄── PUB/SUB orderflow ──►
RTX training
Dashboard + bot
```

- **PyTorch DDP** with **Gloo backend** (TCP/LAN, no NVLink needed)
- Main PC binds ZeroMQ on `tcp://*:5555` (orderflow PUB), `tcp://*:5556` (training batch PUSH), `tcp://*:5557` (control fanout)
- FastAPI on `tcp://*:8100` for control-plane REST: `/training/start`, `/training/status`, `/training/checkpoint`, `/cluster/health`
- Workers fetch model checkpoints from main PC at epoch start; gradients sync via DDP

---

## 4. Phase Roadmap

### Phase 0 — Foundation (Week 1)
DB migration + transport split + distributed training upgrade.

**New files:**
- `src/database/parquet_store.py`
- `scripts/migrate_1sec_to_parquet.py`
- `src/transport/__init__.py`
- `src/transport/zmq_config.py`
- `src/transport/data_bus.py`
- `src/transport/control_api.py`

**Modified files:**
- `src/training/distributed/orchestrator.py` — use `data_bus.push_batch` + `control_api`
- `src/training/distributed/worker.py` — use `data_bus.pull_batch`

### Phase 1 — Level 1 Data Layer (Week 2)
**New:** `src/data_ingestion/orderbook_collector.py`, `src/analysis/orderbook_features.py`, `src/analysis/kalman_smoother.py`
**Modified:** `feature_store.py` (apply Kalman first), `feature_engineering.py` (add OFI/imbalance/microprice), `triple_barrier.py` (causal `t1` audit)

### Phase 2 — Level 2 Alpha Engine (Weeks 3–4)
**New:** `src/analysis/event_time_labeler.py`, `src/models/order_flow_transformer.py`, `src/training/oft_trainer.py` (PurgedKFold + isotonic calibration)
**Modified:** `regime_classifier.py` (BayesianGaussianMixture + partial_fit), `inference_engine.py` (OFT inference path)

### Phase 3 — Level 3 Execution & Simulation (Weeks 5–6)
**New:** `src/simulation/synthetic_exchange.py`, `src/simulation/multi_agent_env.py`, `src/models/rl_base.py`, `src/models/rl_execution_sac.py`, `src/models/rl_execution_ppo.py`, `src/analysis/alpha_decay.py`
**Modified:** `market_replay.py` (use synthetic_exchange), `order_manager.py` (RL agent + alpha decay), `orchestrator.py` (joint OFT+RL training loop with `min(-E[PnL] + λ1·CVaR + λ2·ImpactCost + λ3·InventoryRisk)`)

**RL failover logic:** SAC default. Switch to PPO if last 100 trades' Sharpe < 0 OR action variance spikes 3σ above training mean. Dashboard toggle for manual override.

### Phase 4 — Level 4 Portfolio Optimization (Week 7)
**New:** `src/analysis/cvar_optimizer.py` (CVXPY), `src/analysis/dynamic_threshold.py`
**Modified:** `risk_manager.py` (CVaR-driven sizing), `kelly_criterion.py` (becomes weight prior), `main.py` (dynamic threshold replaces fixed `SIGNAL_THRESHOLD`)

### Phase 5 — Level 5 Institutional Safeguards (Week 8)
**New:** `src/analysis/slippage_model.py`, `src/analysis/beta_neutrality.py`
**Modified:** `order_manager.py` (circuit breakers — max DD, API latency >500ms, data feed inconsistency), `risk_agent.py` (beta_neutrality pre-trade gate), `feature_store.py` (slippage applied to backtest PnL)

### Phase 6 — Dashboard Rework (parallel with Phases 2–5)
**Modified:** `src/dashboard/app.py` — full rework, 8 tabs, JSON-first routes, WebSocket push for live tabs.

**Tabs:**
1. Portfolio — CVaR, beta exposure, net delta, kill switch, breaker status
2. Alpha Engine — OFT signal + confidence, regime state, alpha decay, model version selector
3. Order Flow — live L2 imbalance, OFI, microprice vs mid, symbol selector
4. Risk — daily drawdown gauge, correlation matrix, open inventory, manual flatten
5. Training — cluster status (both PCs, GPU util), walk-forward fold results, calibration curve, trigger run
6. Simulation — adversarial sim PnL, slippage model chart, scenario selector
7. Data — QuestDB row counts, Parquet store size, freshness per symbol, trigger Parquet export, stack version indicator
8. Strategies — existing per-strategy PnL, enable/disable per strategy

---

## 5. Workflow Rules (from CLAUDE.md)

- Approval gate: present plan, wait for explicit approval before any code changes.
- After every change: update `tests/test_dashboard.py` and run; require 0 failures.
- After every completed task: run `restart_all.ps1` so live bot + dashboard reflect latest code.
- Gemini fallback chain: `gemini-3.1-pro-preview` first.
- Testnet by default; never switch to Mainnet without explicit instruction.
- All cache/temp on D: drive only.

---

## 6. Estimated Timeline

8 weeks total. Phase 0 is the critical path — Phases 1+ depend on the data store and transport being in place. Phases 2–5 can overlap with Phase 6 dashboard work.
