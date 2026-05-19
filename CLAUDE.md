> **Inherits global rules from `D:\test 2\CLAUDE.md`** — approval gate (GO/ГО only), no-guessing, regression tests, git lifecycle, D:-drive-only policy.
> **MANDATORY**: Update the "Current System State" section of this file after EVERY change (code, data, models, config). This file is the single source of truth for every new session.

---

# AI Trading Assistance — Complete Project Context

## Quick Start

```powershell
cd "D:\test 2\AI trading assistance"
.\restart_all.ps1          # start everything
.\stop_all.ps1             # stop everything
python -m pytest tests/test_dashboard.py -v   # regression suite (must be 0 failures)
```

- Dashboard: http://localhost:5000
- Monitor: http://localhost:5001
- Cluster/Training: http://localhost:7700

---

## Architecture

- **Trading engine**: `src/main.py` — strictly sequential bot loop, no parallelism
- **Dashboard**: `src/dashboard/app.py` (Flask, port 5000), templates at `src/dashboard/templates/index.html`
- **DB**: ParquetClient (DuckDB + partitioned Parquet) at `data/db/`. No daemon. All DuckDB connections must set `temp_directory='D:/test 2/AI trading assistance/data/cache/duckdb_temp'`
- **Historical OHLCV**: `parquet_store.py` reads from `data/parquet/` (48 GB)
- **State files**: all JSON I/O via `src/utils/safe_json.py` (filelock atomic writes)
- **Constants**: `src/utils/config.py`
- **Model artifacts**: `models/` — joblib + meta JSON pairs

### Ports
| Port | Service | Log |
|------|---------|-----|
| 5000 | Flask Dashboard | logs/dashboard.log |
| 5001 | Monitor Server | logs/monitor.log |
| 7700 | Cluster Training Orchestrator | logs/cluster.log |

---

## Trainers — Model × Timeframe Matrix

| Trainer | Model key | Applicable TFs | Skip TFs | Type |
|---------|-----------|---------------|----------|------|
| `src/engine/trainers/train_base.py` | `base` | 15m, 1h, 4h | 1m, 1d, 1mo | HistGBT tabular |
| `src/engine/trainers/train_trend.py` | `trend` | 15m, 1h, 4h, 1d | 1m, 5m, 1w, 1mo | HistGBT tabular |
| `src/engine/trainers/train_futures.py` | `futures` | 15m, 1h, 4h | 1m, 5m, 1mo | HistGBT tabular |
| `src/engine/trainers/train_scalping.py` | `scalping` | 1m, 5m | 15m+ | HistGBT + SMOTE |
| `src/engine/trainers/train_meta.py` | `meta` | 15m, 1h, 4h | 1m, 1d, 1w, 1mo | HistGBT meta-labeler |
| `src/engine/trainers/train_tft.py` | `tft` | 1h, 4h | 1m, 5m, 15m, 1w, 1mo | Neural DARTS/TFT |
| `src/engine/trainers/train_oft.py` | `oft` | all | — | PyTorch OFT |
| `src/engine/trainers/train_regime.py` | `regime` | 1h (fixed) | — | GMM regime |

Source of truth for (model × TF) rules: `data/training_rules.json` — READ ON EVERY TRAINING STARTUP.

---

## Models in production (`models/`)

```
base_{15m,1h,4h,1d}_model.joblib + _meta.json       trained 2026-05-17
futures_{15m,1h,4h,1d}_model.joblib + _meta.json    trained 2026-05-14–17
meta_{15m,1h,4h,1d}_model.joblib + _meta.json       trained 2026-05-15–17
meta_labeler.joblib + meta_labeler_meta.json
scalping_1m_model.joblib + scalping_1m_meta.json
btc_rf_model.joblib + btc_rf_model_meta.json        (legacy, kept for compat)
regime_classifier.joblib + _meta.json
tft_model.pt                                         Temporal Fusion Transformer
oft_model.pt                                         Order Flow Transformer
manifest.json                                        checksums + versions
_baseline_2026-05-16/                                baseline snapshot
```

---

## Data Ingestion Scripts

| Script | Downloads | Storage |
|--------|-----------|---------|
| `binance_sync.py` | OHLCV via archive + REST top-up | data/parquet/_OHLCV/ |
| `binance_archive_downloader.py` | Monthly ZIP archives (data.binance.vision) | data/parquet/_OHLCV/ |
| `realtime_db_writer.py` | Binance WebSocket live | data/parquet/_OHLCV/ |
| `funding_rate_downloader.py` | Funding rates Bybit/Binance | data/parquet/_FUNDING/ |
| `liquidation_downloader.py` | Liquidation events (Coinglass — needs COINGLASS_API_KEY) | data/parquet/_LIQUIDATIONS/ |
| `open_interest_downloader.py` | OI delta + funding pressure | data/parquet/_OI/ |
| `fear_greed_downloader.py` | Fear & Greed Index | data/parquet/_FEAR_GREED/ |
| `orderbook_collector.py` | L2 snapshots (BTC/ETH/SOL, 100ms) | data/parquet/_L2/ |
| `historical_backfill.py` | Gap fill OHLCV | data/parquet/_OHLCV/ |

Multi-source governance: `src/data_governance/orchestrator.py` (Bybit, CoinGecko, Coinglass, Glassnode, NewsAPI, Reddit, Santiment, Etherscan, FRED, YouTube).

---

## Key Files — Go Directly, Don't Grep

### Phase 1 fixes (COMPLETED 2026-05-18)

| Fix | File | Status |
|-----|------|--------|
| `embargo_td: pd.Timedelta` param | `src/utils/purged_kfold.py` | ✅ Done |
| `embargo_td=pd.Timedelta(hours=48)` (base) | `src/engine/train_model.py` | ✅ Done |
| `embargo_td=pd.Timedelta(hours=24)` (futures) | `src/engine/train_futures_model.py` | ✅ Done |
| `embargo_td=pd.Timedelta(hours=24)` (meta) | `src/engine/train_meta_labeler.py` | ✅ Done |
| `KPIGateFailure` + `hard_gate_wf()` | `src/engine/kpi_gate.py` | ✅ Done |
| WF gate call in base/futures trainers | `train_model.py`, `train_futures_model.py` | ✅ Done |
| Dashboard WF accuracy display | `app.py`, `index.html` | ✅ Done |

### Phase 2 (COMPLETED 2026-05-18)

| Task | File | Status |
|------|------|--------|
| `add_symbol_id()` + `add_explicit_regime()` | `src/analysis/feature_engineering.py` | ✅ Done |
| `symbol_id` + regime columns in FEATURE_COLUMNS | All 5 trainers | ✅ Done |
| `symbol_id` + `optimal_threshold` at inference | `src/analysis/ml_predictor.py` | ✅ Done |
| AFML weights (`compute_afml_weights`) | `src/utils/sample_weights.py` (new) | ✅ Done |
| AFML weights in fold + final model | base, futures, trend, meta trainers | ✅ Done |
| AFML weights skipped (SMOTE incompatible) | scalping trainer | ✅ Done (by design) |
| `find_optimal_threshold()` | `src/utils/threshold_optimizer.py` (new) | ✅ Done |
| `optimal_threshold` + `optimal_sortino` in meta JSON | All 5 trainers | ✅ Done |
| `symbols_sorted` list in meta JSON | All 5 trainers | ✅ Done |
| Symbol + `symbols_sorted` loaded at inference | `ml_predictor.py`, `multi_tf_predictor.py` | ✅ Done |
| `symbol=symbol` passed from main loop | `src/main.py` (4 call sites) | ✅ Done |

### Phase 3 (COMPLETED 2026-05-18)

| Task | File | Status |
|------|------|--------|
| `add_explicit_regime(df)` (bull/bear/chop/high_vol) | `src/analysis/feature_engineering.py` | ✅ Done |
| Regime columns in FEATURE_COLUMNS | All 5 trainers | ✅ Done |
| Regime computed at inference | `src/analysis/ml_predictor.py` | ✅ Done |

### Phase 4 (COMPLETED 2026-05-18)

| Task | File | Status |
|------|------|--------|
| `ChampionRegistry` | `src/engine/champion_challenger.py` (new) | ✅ Done |
| Registry file | `data/champion_registry.json` (auto-created) | ✅ Done |
| `register_challenger()` in all 5 trainers | All trainers after `write_json` | ✅ Done |

---

## Process Management

`restart_all.ps1` starts 13 detached processes (WMI-spawned, survive parent exit):

```
startup_recovery → monitor_server(:5001) → cluster_orchestrator(:7700)
→ dashboard(:5000) + bot → watchlist_downloader → realtime_db_writer
→ data_governance_orchestrator → orderbook_collector + orderbook_parquet_writer
→ debug_supervisor → dashboard_watchdog → training_sweep_watchdog
```

Logs: `logs/bot.log`, `logs/dashboard.log`, `logs/cluster.log`, `logs/data_orchestrator.log`, `logs/realtime_db.log`, `logs/orderbook_collector.log`, `logs/debug_supervisor.log`, `logs/dashboard_watchdog.log`, `logs/training_sweep_watchdog.log`

State files: `data/process_ids.json`, `data/process_deaths.json`, `data/error_state.json`

---

## Feature Engineering (`src/analysis/`)

34 modules. Key ones for training:
- `feature_engineering.py` — `add_rsi`, `add_macd`, `add_bollinger_bands`, `add_roc`, `add_time_features`, `add_taker_and_trade_features`, `add_ofi`, `add_vwap`, `add_funding_zscore`, `add_liquidity_proximity`, `add_atr`, `add_news_sentiment`, `add_l2_features`, `add_tick_features`
- `fractional_diff.py` — `add_fractional_diff(df, d=0.4)` — CAUSAL, no look-ahead
- `triple_barrier.py` — `triple_barrier_labels_vectorized(df, pt=4.0, sl=2.0, max_bars=24)` — labels only, intentional look-ahead
- `regime_classifier.py` — `RegimeClassifier` (GMM, 3 labels: RANGING/TRENDING/VOLATILE) — used by meta-labeler, do NOT modify
- `ml_predictor.py` — single model inference, reads `optimal_threshold` from meta JSON (Phase 2)
- `multi_tf_predictor.py` — multi-timeframe ensemble

FEATURE_COLUMNS (43 features as of 2026-05-18): frac_diff_d40, volatility, dist_sma_7/30, rsi_14, macd/hist, volume_momentum, stoch_k, return_lag1-5, atr_pct, taker_buy_ratio, avg_trade_size, hour, day_of_week, roc_3/7/14, bb_pb, news_sentiment, ofi_z, vwap_dist, liq_proximity, trend_strength, vol_regime, is_trending, is_volatile, signal_rsi/macd/bb, oi_change_pct, fear_greed_norm, liq_long_z, liq_short_z, liq_dom, **symbol_id**, **regime_bull**, **regime_bear**, **regime_chop**, **regime_high_vol** (new Phase 2/3)

---

## Utils (`src/utils/`)

21 modules. Critical ones:
- `purged_kfold.py` — AFML walk-forward CV with embargo + t1-span purging; `embargo_td: pd.Timedelta` param (fixed 2026-05-18)
- `kpi_gate.py` — `evaluate_run()`, `_check_thresholds()`, `hard_gate_wf()`, `KPIGateFailure` — auto-retire on 3 failures (fixed 2026-05-18)
- `sample_weights.py` — `compute_afml_weights(y, t1, returns)` — AFML uniqueness × event-strength × class-balance (new 2026-05-18)
- `threshold_optimizer.py` — `find_optimal_threshold(model, X_cal, y_cal, returns_cal)` → `(best_thr, best_sortino)` (new 2026-05-18)
- `champion_challenger.py` — `ChampionRegistry` backed by `data/champion_registry.json`; PROMOTION_DELTA=0.005 (new 2026-05-18)
- `model_paths.py` — `artifact_paths(model_key, tf)` returns paths for joblib + meta JSON
- `safe_json.py` — atomic JSON writes (filelock)
- `gpu_classifier.py` — `make_classifier()` returns XGBoost-CUDA or HistGBT fallback
- `training_progress.py` — live progress tracking for dashboard
- `model_metrics.py` — compute accuracy, Sharpe, max DD, overfit ratio

---

## Risk (`src/risk/`)

6 modules:
- `live_perf_monitor.py` — rolling 50-trade win rate vs baseline, state at `data/risk/live_perf_state.json`
- `drift_monitor.py` — hourly PSI drift, state at `data/risk/drift_state.json`
- `drift_psi.py` — Population Stability Index calculator
- `drift_baseline.py` — save/load feature distribution baseline
- `kill_switch.py` — circuit breaker (max loss, max DD, consecutive losses)
- `validators.py` — output validation

---

## Tests (`tests/`) — 59 files

Canonical regression suite: `tests/test_dashboard.py` — **0 failures required before any push or deploy**.

Key test files:
- `test_dashboard.py` — main regression suite, includes all phase tests
- `test_purged_kfold.py` — CV validation
- `test_triple_barrier_edge_cases.py` — label integrity
- `test_kpi_gate.py` — KPI threshold gates
- `test_drift_monitor.py` — drift detection
- `test_kill_switch.py` — circuit breaker

---

## Watchlist (20 symbols)

`data/watchlist.json`:
BTC/USDT, SOL/USDT, ADA/USDT, ETH/USDT, BNB/USDT, XRP/USDT, DOGE/USDT, TRX/USDT, AVAX/USDT, SHIB/USDT, DOT/USDT, LINK/USDT, NEAR/USDT, UNI/USDT, LTC/USDT, APT/USDT, ATOM/USDT, HBAR/USDT, ICP/USDT, SUI/USDT

---

## Defaults (never override silently)

- **Testnet only** — never switch to Mainnet without explicit operator instruction
- **DuckDB temp**: `temp_directory='D:/test 2/AI trading assistance/data/cache/duckdb_temp'`
- **Gemini fallback chain** starts with `gemini-3.1-pro-preview`
- **Bot is personal use only** — no Stripe / marketplace / multi-tenant features
- **PT=4×ATR, SL=2×ATR** — do NOT change, this is strategy design. Fix class imbalance via ML weights (Phase 2)

---

## Dashboard API — 95+ endpoints (key groups)

Health: `GET /api/state`, `/api/ai_status`, `/api/monitor/health`, `/api/monitor/services`
Control: `POST /api/control`, `/api/system/restart_all`, `/api/processes/*`
Models: `GET /api/models`, `POST /api/training/run/<key>`, `/api/training/run/all`, `GET /api/training/progress`
Risk: `GET /api/risk/kill_switch/status`, `/api/drift/state`, `/api/live_perf/state`
Data: `GET /api/db/status`, `POST /api/db/query`
Strategy: `GET /api/strategy/full`, `/api/backtest/summary`

---

## ⚡ Current System State
> Update this section after every session

### Data (last updated 2026-05-18)
| Source | Status | Notes |
|---|---|---|
| OHLCV BTC/SOL/ADA/others | ✅ fresh | — |
| OHLCV ETH/BNB/XRP/DOGE/LINK/TRX | ❌ stale (2024-12-31 on 4h/5m/15m) | REST top-up needed |
| Funding rates | ⚠️ ~22 days stale | run funding_rate_downloader |
| CoinGlass v4 — macro | ✅ downloaded 2026-05-18 | `data/coinglass/macro/` (8 files: fear_greed, btc_dominance, stablecoin_mcap, ahr999, puell_multiple, golden_ratio, etf_btc/eth_flow) |
| CoinGlass v4 — futures | ✅ downloaded 2026-05-18 | `data/coinglass/futures/<SYM>/` — 266 files, 20 symbols × OI/LS/FR/Liq/Taker |
| CoinGlass v4 — spot taker | ✅ downloaded 2026-05-18 | `data/coinglass/spot/<SYM>/` — 40 files |
| CoinGlass API key | ✅ set | `COINGLASS_API_KEY` in .env (STARTUP plan, 180d history on 1h/4h) |

Refresh CoinGlass monthly (STARTUP plan, expires after ~1 month):
```powershell
# Futures + spot (all 20 symbols)
python -m src.data_ingestion.coinglass_downloader --futures-only
# Macro only
python -m src.data_ingestion.coinglass_downloader --macro-only
```

Fix stale OHLCV:
```powershell
python -m src.data_ingestion.binance_sync --skip-archive --symbols ETH/USDT BNB/USDT XRP/USDT DOGE/USDT LINK/USDT TRX/USDT --tfs 4h 5m 15m
python -m src.data_ingestion.migrate_csv_to_parquet
python -m src.data_ingestion.funding_rate_downloader
```

### CoinGlass feature integration (completed 2026-05-18)
- `src/data_ingestion/coinglass_downloader.py` — NEW: downloads OI, L/S, funding, liquidations, taker flow, coinbase premium, macro indices
- `src/analysis/feature_engineering.py` — `add_coinglass_features(df, symbol, timeframe)` merges 15 CG columns via `merge_asof`, stable-schema (fills 0.0 if data absent)
- All 5 trainers (`train_model`, `train_futures_model`, `train_trend_model`, `train_scalping_model`, `train_meta_labeler`) call `add_coinglass_features()` and have 10-14 new CG columns in FEATURE_COLUMNS
- `src/analysis/ml_predictor.py` — `_build_all_features(df, symbol="", timeframe="1h")` now also calls `add_coinglass_features`; live predict path gets macro features (fear_greed, btc_dominance, stablecoin_mcap) even without symbol

### Model accuracy (last checked 2026-05-18)
| Model | In-sample | WF | Status |
|---|---|---|---|
| base_1h | 71.6% | 28.7% | ❌ needs retrain with CG features |
| futures_15m | 78.2% | 22.7% | ❌ needs retrain |
| trend_1h | 65.0% | 50.4% | ⚠️ borderline |
| meta_1m | 69.0% | 68.7% | ✅ OK |
| meta_1d | 71.0% | 57.5% | ✅ OK |

Dashboard now shows both WF accuracy and in-sample accuracy per model (Phase 1.1 done).

### Implementation phases
| Phase | Status | Blocker |
|---|---|---|
| Блок 0: Data sync | 🔄 CG done; OHLCV stale symbols + funding still pending | — |
| Phase 1.1: Dashboard WF + in-sample accuracy | ✅ Done | — |
| Phase 1.2: kpi_gate.py hard WF gate | ✅ Done 2026-05-18 | — |
| Phase 1.3: purged_kfold embargo param | ✅ Done 2026-05-18 | — |
| Phase 2: Symbol feature + AFML weights + threshold opt | ✅ Done 2026-05-18 | — |
| Phase 3: Regime features (explicit rules) | ✅ Done 2026-05-18 | — |
| Phase 4: Champion/Challenger registry | ✅ Done 2026-05-18 | — |
| **NEXT: Retrain all models** | ❌ Not started | All pre-training prep complete |

### Degradation monitoring (completed)
- ✅ P1 live_perf_monitor.py
- ✅ P2 drift_psi.py + drift_monitor.py
- ✅ P3 overfit_ratio in all trainers + kpi_gate.py
- ✅ P4 per-fold WF slope gate
- ✅ P5 per-strategy regression guard

### VPS Clean-Slate Migration (started 2026-05-20)
**Branch:** `dev/vps-clean-slate` (ahead of main). **VPS:** 5.104.81.27.

| Phase | Status | Notes |
|---|---|---|
| Phase 0: git branch + VPS hardening | ✅ Done 2026-05-20 | UFW (22/5000), fail2ban, SSH key-only. Branch pushed. Baseline 132 passed. |
| Phase 1A–C: secrets / agent_status / WS timeouts | ✅ Done 2026-05-20 | All .env keys verified. agent_status.json reset. ping_timeout 20→60, close_timeout 10→15 at main.py:1437-1438 |
| Phase 2: Upload 51.95 GB data/parquet to VPS | 🔄 In progress | SFTP PID 23508 on local. Phase3 auto-trigger on VPS (PID 44633) monitors for completion |
| Phase 3: Migrate 121 CSV.gz → Parquet | 🔄 Auto (VPS) | Script at /root/phase3_auto.sh — runs automatically after Phase 2 upload stabilizes (~20 min flat) |
| Phase 4: Harden ohlcv_parquet_loader.py | ❌ Not started | Needs Aider + caller audit. Defer to next session. |
| Phase 5: rclone daily cron | ✅ Done 2026-05-20 | Cron: 3 AM UTC sync to gdrive:trading-bot-backup/ (excl parquet, raw_archive, logs). Archive cleanup: 4 AM UTC. |
| Phase 6: Smoke-tests synthetic data | ❌ Not started | Needs Hetzner/Vast.ai + manual |
| Phase 7: Archive training state | ❌ Not started | Stop bot first |
| Phase 8: Retrain all models | ❌ Not started | After Phase 6-7 |

**Completed code changes on dev/vps-clean-slate:**
- Added PLAN_VPS_CLEAN_SLATE.md / PLAN_VPS_CLEAN_SLATE_RU.md (v11) and PLAN_POST_PRODUCTION_TUNING.md
- src/main.py: ping_timeout=60, close_timeout=15 (commit c24b789)
- .gitignore: added *.bak, data/coinglass/, data/risk/live_perf_state.json
- binance_sync.py: tz-naive/aware datetime fix in step_rest_topup

**To check when user wakes up:**
1. `Get-Content "D:\test 2\AI trading assistance\logs\parquet_upload.log" | Select-Object -Last 10` — upload progress
2. `ssh -i ~/.ssh/trading_bot root@5.104.81.27 "cat /root/trading-bot/logs/phase3_auto.log | tail -20"` — Phase 3 status
