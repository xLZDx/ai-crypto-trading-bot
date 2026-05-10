# AI Trading Assistance — Full App Documentation

Last updated: Phase 10. Architecture follows `updated_architecture_plan_en.md`
§1-18. This document is the canonical operator manual — start here when
returning to the project after a break.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Data lifecycle](#3-data-lifecycle)
4. [Component reference](#4-component-reference)
5. [API reference](#5-api-reference)
6. [Operating procedures](#6-operating-procedures)
7. [Configuration & secrets](#7-configuration--secrets)
8. [Tests](#8-tests)
9. [Phase-by-phase summary](#9-phase-by-phase-summary)
10. [Architecture plan §1-18 — wiring status](#10-architecture-plan-1-18--wiring-status)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick start

```powershell
# 0. From a fresh checkout — install deps (D-only cache)
./venv/Scripts/python.exe -m pip install --no-cache-dir -r requirements.txt

# 1. One-time data prep
./venv/Scripts/python.exe scripts/migrate_1sec_to_parquet.py
./venv/Scripts/python.exe scripts/migrate_to_parquet.py --timeframe 1m
./venv/Scripts/python.exe scripts/migrate_to_parquet.py --timeframe 1d
./venv/Scripts/python.exe scripts/migrate_news_to_parquet.py

# 2. Boot the live system (QuestDB + bot + dashboard + realtime + orchestrator)
./restart_all.ps1

# 3. Open the dashboard
# http://127.0.0.1:5000      (live trading + Phase 6 8-tab nav)
# http://127.0.0.1:5001      (monitor / health)

# 4. Train the institutional models (overnight)
./launch_joint_training.ps1
```

---

## 2. Architecture at a glance

```
                         ┌──────────────────┐
                         │   USER (you)     │
                         └────────┬─────────┘
                                  │
                                  ▼
                ┌────────────────────────────────┐
                │ Dashboard (Flask, port 5000)   │
                │ ─ 8 tabs (Phase 6)             │
                │ ─ REAL vs TEST/TRAIN switcher  │
                └────┬───────────────────────┬───┘
                     │                       │
                     │ /api/state            │ /api/decision_summary
                     │ /api/balance/{mode}   │ /api/oft_signal
                     │ /api/news             │ /api/parquet/coverage
                     │ /api/orchestrator     │ /api/rate_limiter/stats
                     ▼                       ▼
        ┌────────────────────┐   ┌────────────────────────┐
        │ Bot main.py        │   │ Data Orchestrator      │
        │ ─ process_kline    │   │ ─ 17 connectors        │
        │ ─ InstitutionalGate│   │ ─ rate-limited HTTP    │
        │ ─ §11-18 wired     │   │ ─ scheduled polls      │
        └─┬─────────────────┬┘   └────────────┬───────────┘
          │                 │                 │
          ▼                 ▼                 ▼
   ┌────────────┐    ┌────────────┐    ┌────────────────┐
   │ Binance    │    │ QuestDB    │    │ Parquet store  │
   │ Spot/Fut   │    │ (hot path) │    │ (cold path)    │
   │ + WS       │    │ ILP :9009  │    │ data/parquet/  │
   └────────────┘    │ REST :9000 │    │  {SYM}/{tf}/   │
                     └────────────┘    │  yyyymm=*/     │
                            ▲          └────────────────┘
                            │
                  Realtime DB Writer
                  (closed bars only, idempotent)
```

**Two storage tiers:**
- **QuestDB** — hot path (recent ticks; ILP writes ~1 M rows/sec)
- **Parquet** — cold path (full history; 49.75 GB / 3.27 B rows of 1-sec data)

**Two processing paths:**
- **Live trading loop** (`src/main.py`) — Phase 10 reads from Parquet (with CSV.gz fallback)
- **Training loop** (`src/training/joint_oft_rl.py`) — reads Parquet exclusively

---

## 3. Data lifecycle

```
┌─ EXTERNAL FEEDS ───────────────────────────────────────┐
│  Binance archive   (data.binance.vision)               │
│  Binance REST      (api.binance.com)                   │
│  Binance WS        (stream.binance.com)                │
│  Bybit / OKX / Coinbase / Kraken (other connectors)    │
│  CryptoCompare news / The Block / Reddit / Telegram    │
│  FRED / CoinGecko / Fear & Greed / DefiLlama           │
└───────────┬───────────┬─────────────────┬─────────────┘
            │           │                 │
            │ archive   │ realtime ws     │ scheduled REST polls
            ▼           ▼                 ▼
     ┌──────────────────────────────────────────────┐
     │ Rate-limited HTTP layer                      │
     │ src/data_ingestion/rate_limiter.py           │
     │ (token bucket per host, weight-aware,        │
     │ reactive 429/418 backoff)                    │
     └─────┬───────────┬─────────────────────┬──────┘
           │           │                     │
           ▼           ▼                     ▼
    data/raw/    QuestDB ILP            QuestDB ILP
   (.csv.gz)    (market_data table)    (model_signals,
                                        news_sentiment, …)
        │              │                       │
        │ migration    │ nightly rollover      │ nightly rollover
        │ scripts      ▼                       ▼
        ▼      ┌────────────────────────────────────┐
     data/parquet/   {SYM}/{tf}/yyyymm=*/data.parquet
                                ▲
                                │ partitioned, zstd-compressed
                                │ Hive layout
        ┌───────────────────────┘
        │ DataLens.training_frame() joins everything
        ▼
   src/analytics/data_lens.py
        │
        ▼
   src/training/joint_oft_rl.py     (training)
   src/main.py                       (live, via feature_reader)
   src/engine/train_model_v2.py      (modernized training)
```

### Migration scripts

| Source | Destination | Script |
|---|---|---|
| `*_spot_1s.csv.gz` (34 GB) | `{SYM}/yyyymm=*/` | [scripts/migrate_1sec_to_parquet.py](scripts/migrate_1sec_to_parquet.py) |
| `*_1m.csv.gz` | `{SYM}/1m/yyyymm=*/` | [scripts/migrate_to_parquet.py](scripts/migrate_to_parquet.py) `--timeframe 1m` |
| `*_1h.csv.gz` | `{SYM}/1h/yyyymm=*/` | same `--timeframe 1h` |
| `*_1d.csv.gz` | `{SYM}/1d/yyyymm=*/` | same `--timeframe 1d` |
| `*_funding.csv.gz` | `{SYM}/funding/yyyymm=*/` | same `--timeframe funding` |
| `cryptocompare_news.csv` | `_NEWS/news/yyyymm=*/` | [scripts/migrate_news_to_parquet.py](scripts/migrate_news_to_parquet.py) |

### Idempotency rules

- Parquet ingest skips months that already have a non-empty file in their partition.
- Archive downloader uses HEAD probe + listing-date cache (`data/binance_listing_dates.json`) to avoid wasted GET-404s.
- Realtime WS writer only emits **closed bars** (`k.x == True`) so QuestDB's primary key (symbol, timeframe, timestamp) deduplicates restarts.
- Startup recovery (`src/data_ingestion/startup_recovery.py`) is safe to re-run anytime.

---

## 4. Component reference

### 4.1 Live bot — `src/main.py` (`MultiAssetTrader`)

| Method | What |
|---|---|
| `__init__` | Spins up: analyzers, predictors, regime classifier, feature store, market makers, mean-reversion, momentum engine, telegram monitor, **InstitutionalGate (Phase 9)**, **dual_balance refresh (Phase 9)**, **β-history attach (Phase 10E)**, **dynamic threshold cache (Phase 10D)**, **alpha-decay tracking (Phase 10B)** |
| `process_kline(symbol, price)` | Per-tick decision loop. Phase 10 reads bars from Parquet via `feature_reader.load_recent_bars` (CSV.gz fallback). Calls `InstitutionalGate.pre_trade_check` and `InstitutionalGate.executed_price` on every order. Closes alpha-decayed positions. |
| `evaluate_all_strategies` | Combines Elliott Wave + ML + OU + momentum + funding-arb + regime-aware filters into a final BUY/SELL/HOLD signal. |
| `_attach_beta_history` | Phase 10E. Builds 180d returns DataFrame from Parquet 1d data and feeds it to the gate's β-neutrality filter so it's non-noop. |
| `_refresh_dynamic_thresholds` | Phase 10D. Hourly refit of per-symbol threshold from recent (probs, returns); blends with regime base. |

### 4.2 Institutional gate — `src/engine/institutional_gate.py`

| Method | Plan § | Live? |
|---|---|---|
| `pre_trade_check(symbol, side, notional, ...)` | §17 + §18 | ✅ |
| `cvar_size(symbols, scenarios, p_wins, capital)` | §13-14 | helper, not called per-tick |
| `best_threshold(probs, returns)` | §15 | ✅ via `_refresh_dynamic_thresholds` |
| `executed_price(mid, side, size, depth)` | §16 | ✅ |
| `should_exit_decay(strength, time)` | §12 | ✅ via alpha-decay loop |
| `attach_beta_filter(history)` | §17 | ✅ via `_attach_beta_history` |

### 4.3 Phase 1-5 modules (built; usage above)

| File | Plan § | Purpose |
|---|---|---|
| `src/analysis/kalman_smoother.py` | §3 | `pykalman` filter for noise-cleaning close prices |
| `src/analysis/orderbook_features.py` | §2 | OFI / imbalance / microprice formulas |
| `src/data_ingestion/orderbook_collector.py` | §1 | L2 stream from Binance public depth feed → ZeroMQ DataBus |
| `src/analysis/event_time_labeler.py` | §5 | Regime-normalized barriers, binary classification labels |
| `src/models/order_flow_transformer.py` | §6 | OFT: Event-Embed → OB-Encoder → Temporal-Transformer → Cross-Attn |
| `src/training/oft_trainer.py` | §7 | PurgedKFold + isotonic calibration + microstructure noise augment |
| `src/analysis/regime_classifier.py` | §8 | BayesianGaussianMixture (DP prior) + warm-start `partial_fit` |
| `src/simulation/synthetic_exchange.py` | §9 | Differentiable matching engine (softmax fill) |
| `src/simulation/multi_agent_env.py` | §9 | OFT-alpha vs noise/momentum baselines self-play |
| `src/training/joint_oft_rl.py` | §10 | OFT supervised then SAC inside SyntheticExchange |
| `src/models/rl_base.py / rl_execution_sac.py / rl_execution_ppo.py` | §11 | SAC primary, PPO backup. Reward = `PnL - λ·inventory²` |
| `src/analysis/alpha_decay.py` | §12 | `signal * exp(-decay_rate * t)` |
| `src/analysis/cvar_optimizer.py` | §13-14 | CVXPY Rockafellar-Uryasev LP, risk-parity prior |
| `src/analysis/dynamic_threshold.py` | §15 | Sharpe-grid threshold finder + rolling refit |
| `src/analysis/slippage_model.py` | §16 | Linear + book-walk slippage; `real_price()` formula |
| `src/analysis/beta_neutrality.py` | §17 | OLS β vs factor; `would_breach()` pre-trade gate |
| `src/engine/order_manager.py` | §18 | `circuit_breaker_check` (max DD / latency / staleness) |

### 4.4 Data governance — `src/data_governance/`

```
src/data_governance/
├ __init__.py              public API: REGISTRY, register, list_sources, GovernanceConfig
├ base.py                  DataSourceConnector ABC
├ registry.py              @register decorator
├ config.py                data/data_governance.json reader
├ orchestrator.py          --list / --once / --names runner
└ connectors/
   ├ bybit.py              ✅ free
   ├ okx.py                ✅ free
   ├ coinbase.py           ✅ free
   ├ kraken.py             ✅ free
   ├ coingecko.py          ✅ free
   ├ fear_greed.py         ✅ free
   ├ defillama.py          ✅ free
   ├ theblock_rss.py       ✅ free RSS
   ├ cryptocompare_news.py ✅ free + optional API key
   ├ fred.py               🔑 needs FRED_API_KEY
   ├ glassnode.py          🔑 needs GLASSNODE_API_KEY
   ├ santiment.py          🔑 needs SANTIMENT_API_KEY
   ├ newsapi.py            🔑 needs NEWSAPI_KEY
   ├ coinglass.py          🔑 needs COINGLASS_API_KEY
   ├ etherscan.py          🔑 needs ETHERSCAN_API_KEY + watchlist
   ├ youtube_transcripts.py 🔑 needs data/youtube_watchlist.json
   └ reddit.py             🔑 needs REDDIT_CLIENT_ID/SECRET/USER_AGENT + praw
```

### 4.5 Storage — `src/database/`

| File | Purpose |
|---|---|
| `parquet_store.py` | `ParquetStore.ingest_csv / query / symbol_status / status / list_timeframes` |
| `questdb_client.py` | ILP writes (`write_market_candle`, `write_signal`, `write_news_sentiment`, …) + REST queries |
| `retention_manager.py` | Tracks (symbol, tf, yyyymm) partitions; `mark_trained` / `archive_eligible` / `prune_archived` |
| `google_drive_backup.py` | pydrive2-based archive uploader (graceful no-op when no creds) |
| `schema.py` | QuestDB DDL — runs idempotently from `restart_all.ps1` |

### 4.6 Analytics — `src/analytics/`

| File | Purpose |
|---|---|
| `data_lens.py` | `DataLens.training_frame()` — joins OHLCV + funding + news + macro for any (symbol, timeframe, period) |
| `decision_metrics.py` | `DecisionMetrics.summarize()` — go/no-go bundle with coverage, feature, model_health, risk, execution |

### 4.7 Phase 10 additions

| File | Phase | Purpose |
|---|---|---|
| `src/analysis/feature_reader.py` | 10A | Parquet-first reader with CSV.gz fallback |
| (in `main.py`) `_attach_beta_history` | 10E | Auto-loads β-filter on startup |
| (in `main.py`) `_refresh_dynamic_thresholds` | 10D | Hourly refit of entry threshold |
| (in `main.py`) alpha-decay block in `process_kline` | 10B | Closes positions whose signal has decayed |
| `src/analysis/feature_engineering.py:add_news_sentiment` | 10F | Now reads Parquet first, falls back to CSV |
| `src/engine/train_model_v2.py` | 10 | Modernized trainer using DataLens + event-time labels |

---

## 5. API reference

### Live state (port 5000)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Bot's current_state (signals, balances, regime, quant) |
| GET | `/api/control` | Run-state + selected AI model |
| POST | `/api/control` | Toggle running / set model |
| GET | `/api/trades` | Trade history |
| GET | `/api/logs` | Log tail |
| POST | `/api/chat` | Gemini-backed chat |

### Phase 9 — Dual balance + analytics

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/balance/real` | `data/balance_real.json` (Binance live) |
| GET | `/api/balance/virtual` | `data/balance_virtual.json` (sim/training) |
| POST | `/api/balance/virtual/reset` | Reset virtual balance |
| GET | `/api/news` | Recent news from `_NEWS/news/yyyymm=*` partition |
| GET | `/api/oft_signal/{sym}` | Latest OFT μ/σ/p_move/liquidity_risk |
| GET | `/api/orchestrator/sources` | Registered connectors + metadata |
| GET | `/api/orchestrator/config` | `data/data_governance.json` |
| GET | `/api/retention/stats` | RetentionManager partition stats |
| GET | `/api/rate_limiter/stats` | Per-host token bucket usage |
| GET | `/api/decision_summary/{sym}?tf=1h` | DecisionMetrics summary |
| GET | `/api/parquet/coverage` | ParquetStore status |

### Distributed cluster

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cluster/status` | Worker + task summary |
| POST | `/api/cluster/submit` | Submit a training task |
| POST | `/api/cluster/register` | Worker heartbeat |
| POST | `/api/cluster/task_update` | Worker reports result |

---

## 6. Operating procedures

### 6.1 Daily operations

```powershell
# Start everything (idempotent — safe to re-run)
./restart_all.ps1

# Tail logs
Get-Content logs/trading.log -Tail 50 -Wait

# Stop everything
./stop_all.bat       # if you have one — otherwise kill via process_ids.json
```

### 6.2 Training

**Quick — modernized RF on the new Parquet data:**
```powershell
./venv/Scripts/python.exe -m src.engine.train_model_v2 --symbol BTC/USDT --tf 1h
```

**Overnight — joint OFT + SAC:**
```powershell
./launch_joint_training.ps1
```

**Status — what got produced:**
```powershell
ls models/*.joblib, models/*.pt
cat models/btc_rf_model_meta.json
```

### 6.3 Migration / re-ingest

```powershell
# After downloading fresh Binance archives:
./venv/Scripts/python.exe scripts/migrate_to_parquet.py --timeframe 1h
./venv/Scripts/python.exe scripts/migrate_to_parquet.py --timeframe 1d

# After downloading fresh news:
./venv/Scripts/python.exe scripts/migrate_news_to_parquet.py

# Run gap report:
./venv/Scripts/python.exe scripts/report_data_gaps.py
```

### 6.4 Adding a new data source

1. Drop a new file in `src/data_governance/connectors/<name>.py` subclassing `DataSourceConnector`.
2. Set `META = ConnectorMeta(name=..., host=..., priority=..., category=...)`.
3. Implement `is_available()` and `pull_history()`.
4. Add the file to `src/data_governance/connectors/__init__.py` so it auto-registers.
5. (Optional) Add the host to `_DEFAULT_HOSTS` in `rate_limiter.py`.
6. Add a default `SourceSetting` in `GovernanceConfig.default()` if you want it on by default.
7. Restart — the orchestrator picks it up automatically.

### 6.5 Cleaning up

```powershell
# Audit unused model files (dry-run, then --apply):
./venv/Scripts/python.exe scripts/cleanup_models.py
./venv/Scripts/python.exe scripts/cleanup_models.py --apply

# Reorg legacy 1-sec parquet (after bulk migration was done into legacy layout):
./venv/Scripts/python.exe scripts/reorg_1s_to_subdir.py --dry-run
./venv/Scripts/python.exe scripts/reorg_1s_to_subdir.py
```

---

## 7. Configuration & secrets

All on-disk config is JSON:

| File | Purpose |
|---|---|
| `data/control.json` | Runtime flags (running, selected_model) |
| `data/state.json` | Bot's current_state snapshot |
| `data/strategy_config.json` | Per-strategy enable/disable |
| `data/watchlist.json` | Symbols to trade |
| `data/process_ids.json` | PIDs of managed processes (used by `restart_all.ps1`) |
| `data/data_governance.json` | Per-source enable / priority / poll-interval |
| `data/binance_listing_dates.json` | First-month-known cache (auto-updated) |
| `data/balance_real.json` / `data/balance_virtual.json` | Dual-balance state |
| `data/youtube_watchlist.json` | YouTube video IDs to fetch transcripts |
| `data/etherscan_wallets.json` | ETH wallets to track |
| `data/retention_index.json` | Trained-on partition tracking |

### Environment variables (`.env`)

| Var | Purpose |
|---|---|
| `API_KEY` / `API_SECRET` | Binance Spot credentials |
| `FUTURES_API_KEY` / `FUTURES_API_SECRET` | Binance Futures credentials |
| `USE_TESTNET` | `True` keeps the bot on testnet (default) |
| `GEMINI_API_KEY` | Google Gemini for the LLM veto |
| `FRED_API_KEY` | FRED macro data |
| `CRYPTOCOMPARE_API_KEY` | CryptoCompare news (free higher quota) |
| `COINGLASS_API_KEY` | CoinGlass aggregated funding |
| `GLASSNODE_API_KEY` | Glassnode on-chain |
| `SANTIMENT_API_KEY` | Santiment social/dev |
| `NEWSAPI_KEY` | NewsAPI.org |
| `ETHERSCAN_API_KEY` | Etherscan whale tracker |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` | Reddit PRAW |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | Telegram message stream |
| `GDRIVE_SA_JSON` | Path to Google Drive service-account JSON (for retention archive) |
| `ZMQ_ORDERFLOW_PORT` etc. | Override default ZeroMQ ports |
| `ARCHIVE_MAX_WORKERS` | Override default 8 archive download threads |
| `CONTROL_API_PORT` | Default 8100 |

---

## 8. Tests

```powershell
# Per-phase test suites
./venv/Scripts/python.exe tests/test_phase0.py
./venv/Scripts/python.exe tests/test_phase1.py
./venv/Scripts/python.exe tests/test_phase2.py
./venv/Scripts/python.exe tests/test_phase3.py
./venv/Scripts/python.exe tests/test_phase4.py
./venv/Scripts/python.exe tests/test_phase5.py
./venv/Scripts/python.exe tests/test_phase7.py
./venv/Scripts/python.exe tests/test_phase8.py
./venv/Scripts/python.exe tests/test_phase9.py
./venv/Scripts/python.exe tests/test_phase10.py
./venv/Scripts/python.exe tests/test_multi_timeframe.py

# Full dashboard regression sweep (offline = no live HTTP calls)
./venv/Scripts/python.exe tests/test_dashboard.py --offline
```

Current totals: **>500 dashboard assertions, 9 per-phase suites all green.**

---

## 9. Phase-by-phase summary

| Phase | Title | Highlights |
|---|---|---|
| **0** | Foundation | DuckDB+Parquet store, FastAPI control plane, ZeroMQ data bus |
| **1** | Data Layer | Kalman, OFI/imbalance/microprice, L2 collector, causal audit |
| **2** | Alpha Engine | OFT model, OFT trainer (PurgedKFold + isotonic), BayesianGMM |
| **3** | Execution & RL | Synthetic exchange, SAC + PPO, alpha decay, multi-agent env |
| **4** | Portfolio | CVaR optimizer (CVXPY), risk parity, dynamic Sharpe threshold |
| **5** | Safeguards | Slippage model, β-neutrality, circuit breakers (DD / latency / staleness) |
| **6** | Dashboard rework | 8-tab nav strip, REAL vs TEST/TRAIN switcher, new API routes |
| **7** | Continuous pipeline | realtime_db_writer, startup_recovery, retention_manager, GDrive backup |
| **8** | Data governance | rate_limiter, binance_sync, 11 connectors, archive HEAD-probe |
| **9** | Integration + analytics | InstitutionalGate, DataLens, DecisionMetrics, dual-balance, +6 more connectors |
| **10** | Live wiring | Parquet-first feature reader, β-history attach, dynamic threshold, alpha decay exit, Parquet news, train_model_v2 |

---

## 10. Architecture plan §1-18 — wiring status

| § | Item | Built | Wired in `main.py` |
|---|---|---|---|
| 1 | Microstructure data collection | ✅ | partial (collector runs separately) |
| 2 | OFI / imbalance / microprice | ✅ | ✅ via FeatureStore |
| 3 | Kalman filter | ✅ | ✅ via FeatureStore |
| 4 | Causal feature audit | ✅ | training-time only (intended) |
| 5 | Event-time labeling | ✅ | training-time only (intended) |
| 6 | Order Flow Transformer | ✅ | ✅ inference path live (waits on checkpoint) |
| 7 | OFT training methodology | ✅ | training-time only (intended) |
| 8 | Bayesian regime model | ✅ | ✅ — used by `evaluate_all_strategies` |
| 9 | Synthetic adversarial sim | ✅ | training-time only (intended) |
| 10 | Joint OFT+RL training | ✅ | training-time only (intended) |
| 11 | HFT inventory hedging | ✅ | ✅ in RL reward shaping |
| 12 | Alpha decay model | ✅ | **✅ Phase 10B** |
| 13 | CVaR optimizer | ✅ | helper available; portfolio-level |
| 14 | Risk parity / confidence | ✅ | helper available |
| 15 | Dynamic threshold | ✅ | **✅ Phase 10D** |
| 16 | Slippage / execution cost | ✅ | ✅ Phase 9 |
| 17 | Beta neutrality | ✅ | **✅ Phase 10E** |
| 18 | Circuit breakers | ✅ | ✅ Phase 9 |

---

## 11. Troubleshooting

### Bot won't start
1. Check `data/state.json` — startup errors are logged there.
2. Check `logs/trading.log` for the traceback.
3. `restart_all.ps1` will skip QuestDB if Docker is missing — bot keeps running but `inference_engine` has no model_signals table.

### Realtime WS writer not receiving bars
- Symptom: `data/parquet/{SYM}/1m/yyyymm=*/` doesn't grow.
- Check `logs/realtime_db.log` for connection errors.
- Bars only arrive at *bar close* (60s for 1m, 3600s for 1h).

### Archive downloader 404 storm
- Pre-Phase 8: this was the bottleneck (~595 wasted GETs in one run).
- Phase 8 fix: HEAD probe + listing-date cache. If still slow, set `ARCHIVE_MAX_WORKERS=16`.

### OFT inference returns `available: false`
- Run `./launch_joint_training.ps1` to produce `models/oft_model.pt`.
- Verify with `ls models/oft_model.pt`.

### β-neutrality blocks every trade
- Check `data/parquet/{SYM}/1d/` has data — β fitting needs ≥ 100 daily bars.
- Increase cap: edit `_attach_beta_history` in `main.py`, raise `max_beta_exposure`.

### Dashboard tab is blank
- Check the matching `/api/...` endpoint with `curl` — error JSON tells you which underlying module failed.
- Phase 6 panes auto-refresh every 30 s.

### Disk filling up
```
data/parquet/    49 GB cold history
data/raw/        59 GB original CSVs (deletable per Option A)
data/cache/      ~13 GB DuckDB temp during ingest (clears after run)
```
- Run `./venv/Scripts/python.exe scripts/cleanup_models.py --apply` to archive unused checkpoints.
- See [the CSV deletion thread](APP_DOCUMENTATION.md#csv-deletion) — Option A reclaims 34 GB safely.

---

*End of documentation. Update this file whenever a phase lands.*
