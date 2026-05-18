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

### Phase 1 fixes (NOT YET IMPLEMENTED as of 2026-05-18)

| Bug | File | Line | Fix |
|-----|------|------|-----|
| Embargo too small (multi-symbol) | `src/utils/purged_kfold.py` | 74 | Add `embargo_td: pd.Timedelta` param |
| Embargo wrong (base trainer) | `src/engine/train_model.py` | 453 | `embargo_td=pd.Timedelta(hours=48)` |
| Embargo wrong (futures trainer) | `src/engine/train_futures_model.py` | 209 | `embargo_td=pd.Timedelta(hours=24)` |
| Embargo wrong (meta trainer) | `src/engine/train_meta_labeler.py` | 328 | `embargo_td=pd.Timedelta(hours=24)` |
| WF gate missing | `src/engine/kpi_gate.py` | — | Add `KPIGateFailure` + `hard_gate_wf(wf_acc, model_key, threshold=0.50)` |
| WF gate call (base) | `src/engine/train_model.py` | after 490 | `hard_gate_wf(_wf_mean_acc, 'base')` |
| WF gate call (futures) | `src/engine/train_futures_model.py` | after 243 | same |
| Dashboard shows wrong accuracy | `src/dashboard/app.py` | 362 | `.get('walk_forward_mean_acc') or .get('accuracy')` |
| Dashboard shows wrong accuracy | `src/dashboard/app.py` | 2278–2321 | `headline_acc = wf_pct` not `acc_pct` |
| Dashboard shows wrong accuracy | `src/dashboard/templates/index.html` | 3966–3967, 4168–4169 | `modelMeta.walk_forward_mean_acc ?? modelMeta.accuracy` |
| Dashboard label wrong | `src/dashboard/templates/index.html` | 1692 | `◆ ACCURACY` → `◆ WF ACC` |
| Dashboard model card wrong | `src/dashboard/templates/index.html` | 6675 | `m.accuracy_walk_forward ?? m.accuracy` |

### Phase 2 (NOT YET IMPLEMENTED)

| Task | File | Note |
|------|------|------|
| Symbol feature | `src/analysis/feature_engineering.py` | Add `add_symbol_id(df, symbol, known_symbols)` |
| Symbol in FEATURE_COLUMNS | All 5 trainers | Add `'symbol_id'` |
| Symbol at inference | `src/analysis/ml_predictor.py` | Read map from meta JSON |
| AFML weights | New `src/utils/sample_weights.py` | `compute_afml_weights(y, t1, returns)` |
| Replace balanced weights | All 5 trainers | Replace `compute_sample_weight('balanced', y)` |
| Threshold optimizer | New `src/utils/threshold_optimizer.py` | Extracted from `train_meta_labeler.py` lines 68–91 |
| Save optimal_threshold | 4 trainers (base/futures/trend/scalping) | Add to meta JSON |
| Use optimal_threshold | `src/analysis/ml_predictor.py` | Instead of fixed 0.5 |

### Phase 3 (NOT YET IMPLEMENTED)

| Task | File |
|------|------|
| Regime detection (explicit rules) | `src/analysis/regime_classifier.py` — add `compute_explicit_regime(df)` |
| Regime features | `src/analysis/feature_engineering.py` — add `add_explicit_regime(df)` |
| BTC dominance + market cap | New `src/data_ingestion/global_market_downloader.py` |
| Global market features | `src/analysis/feature_engineering.py` — add `add_global_market_features()` |

### Phase 4 (NOT YET IMPLEMENTED)

| Task | File |
|------|------|
| Trading metrics | New `src/utils/trading_metrics.py` — Sharpe, PF, Expectancy, MaxDD |
| Champion/Challenger | New `src/engine/champion_challenger.py` |
| Registry file | `data/champion_registry.json` |
| Inference routing | `src/analysis/ml_predictor.py` — `get_active_artifact()` |

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

FEATURE_COLUMNS (38 features, defined in `train_model.py` lines 143–173): frac_diff_d40, volatility, dist_sma_7/30, rsi_14, macd/hist, volume_momentum, stoch_k, return_lag1-5, atr_pct, taker_buy_ratio, avg_trade_size, hour, day_of_week, roc_3/7/14, bb_pb, news_sentiment, ofi_z, vwap_dist, liq_proximity, trend_strength, vol_regime, is_trending, is_volatile, signal_rsi/macd/bb, oi_change_pct, fear_greed_norm, liq_long_z, liq_short_z, liq_dom

---

## Utils (`src/utils/`)

21 modules. Critical ones:
- `purged_kfold.py` — AFML walk-forward CV with embargo + t1-span purging (⚠️ embargo bug at line 74)
- `kpi_gate.py` — `evaluate_run()`, `_check_thresholds()` — auto-retire on 3 failures (⚠️ `KPIGateFailure` not yet added)
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
| Phase 1.2: kpi_gate.py hard WF gate | ❌ Not started | — |
| Phase 1.3: purged_kfold embargo param | ❌ Not started | — |
| Phase 2: Symbol feature + AFML weights + threshold opt | ❌ Not started | Needs Phase 1 done |
| Phase 3: Regime + global market features | ❌ Not started | — |
| Phase 4: Trading metrics + Champion/Challenger | ❌ Not started | — |

### Degradation monitoring (completed)
- ✅ P1 live_perf_monitor.py
- ✅ P2 drift_psi.py + drift_monitor.py
- ✅ P3 overfit_ratio in all trainers + kpi_gate.py
- ✅ P4 per-fold WF slope gate
- ✅ P5 per-strategy regression guard
