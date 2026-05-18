import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from src.data_ingestion.ohlcv_parquet_loader import load_ohlcv
from sklearn.ensemble import HistGradientBoostingClassifier  # kept for type compat
from sklearn.calibration import CalibratedClassifierCV
# 2026-05-10 GPU migration: tabular trainers now go through make_classifier
# which returns XGBoost-on-CUDA when GPU is available, sklearn HistGBT
# fallback otherwise. Worker is configured via dual-lane spawn so the
# cpu-lane worker has CUDA_VISIBLE_DEVICES='' and silently uses HistGBT.
from src.utils.gpu_classifier import make_classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from src.utils.purged_kfold import PurgedKFold
from src.engine.kpi_gate import hard_gate_wf, KPIGateFailure
from src.utils.sample_weights import compute_afml_weights
from src.utils.threshold_optimizer import find_optimal_threshold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_base')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features,
    add_ofi, add_vwap, add_funding_zscore, add_liquidity_proximity, add_atr,
    add_news_sentiment, add_coinglass_features, add_explicit_regime,
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
from src.data_ingestion.open_interest_downloader import load_open_interest
from src.data_ingestion.fear_greed_downloader import load_fear_greed
from src.data_ingestion.liquidation_downloader import load_liquidations


_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'training_rules.json',
)

# (model_key, param_key) → (expected_type, min_val, max_val)
# min/max only checked for numeric types.
_HP_SCHEMA: dict = {
    'n_estimators': (int, 1, 10_000),
    'max_depth':    (int, 1, 50),
    'class_weight': (str, None, None),
}
_HP_DEFAULTS: dict = {'n_estimators': 500, 'max_depth': 8, 'class_weight': 'balanced'}


def _load_model_params(model_key: str) -> tuple:
    """Load and validate HP from training_rules.json for *model_key*.

    Returns:
        (params_dict, rules_version, params_hash_hex16)
        Falls back to _HP_DEFAULTS on any error; warns explicitly in every case.
    """
    import hashlib

    def _warn_fallback(reason: str):
        log.warning("HP load: %s -- using defaults %s", reason, _HP_DEFAULTS)

    try:
        with open(_RULES_PATH, 'r', encoding='utf-8') as fh:
            rules = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _warn_fallback(f"cannot read training_rules.json ({exc})")
        return dict(_HP_DEFAULTS), None, None

    rules_version = rules.get('_version')
    entry = rules.get('models', {}).get(model_key, {})

    if 'params' not in entry:
        _warn_fallback(f"model '{model_key}' has no 'params' key in training_rules.json")
        return dict(_HP_DEFAULTS), rules_version, None

    raw_params = entry['params']
    validated: dict = {}

    for key, (expected_type, lo, hi) in _HP_SCHEMA.items():
        if key not in raw_params:
            log.warning("HP load: '%s' missing for model '%s'; using default %s=%s",
                        key, model_key, key, _HP_DEFAULTS[key])
            validated[key] = _HP_DEFAULTS[key]
            continue

        val = raw_params[key]
        if not isinstance(val, expected_type):
            log.warning("HP load: '%s' expected %s got %s; using default",
                        key, expected_type.__name__, type(val).__name__)
            validated[key] = _HP_DEFAULTS[key]
            continue

        if lo is not None and not (lo <= val <= hi):
            log.warning("HP load: '%s'=%s out of range [%s, %s]; using default",
                        key, val, lo, hi)
            validated[key] = _HP_DEFAULTS[key]
            continue

        validated[key] = val

    # ── CIO overrides MERGE (post-validation, schema-bounded) ──
    # If the operator promoted a CIO Agent proposal, `cio_overrides` lives at
    # rules.models.<key>.cio_overrides. Merge into validated HPs ONLY for
    # keys that pass the schema check — drops anything that wouldn't survive
    # the standard validation loop. Audit-only keys (`_applied_at` etc.) are
    # already stripped by load_cio_overrides.
    cio_block = entry.get('cio_overrides') or {}
    cio_merged: dict = {}
    if cio_block:
        from src.utils.cio_overrides import load_cio_overrides as _load_cio
        clean = _load_cio(model_key)
        for key, (expected_type, lo, hi) in _HP_SCHEMA.items():
            if key not in clean:
                continue
            val = clean[key]
            if not isinstance(val, expected_type):
                log.warning("[CIO override] '%s' expected %s got %s -- skipping",
                            key, expected_type.__name__, type(val).__name__)
                continue
            if lo is not None and not (lo <= val <= hi):
                log.warning("[CIO override] '%s'=%s out of range [%s, %s] -- skipping",
                            key, val, lo, hi)
                continue
            cio_merged[key] = val
            validated[key] = val
        if cio_merged:
            log.info("[CIO override] merged into %s params: %s", model_key, cio_merged)

    params_hash = hashlib.sha256(
        json.dumps(validated, sort_keys=True).encode()
    ).hexdigest()[:16]

    log.info("HP loaded from training_rules.json (version=%s, hash=%s): %s",
             rules_version, params_hash, validated)
    return validated, rules_version, params_hash


FEATURE_COLUMNS = [
    'frac_diff_d40',        # fractional diff replaces raw return
    'volatility',
    'dist_sma_7', 'dist_sma_30',
    'rsi_14',
    'macd', 'macd_hist',
    'volume_momentum',
    'stoch_k',
    'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
    'atr_pct',
    'taker_buy_ratio', 'avg_trade_size',
    'hour', 'day_of_week',
    'roc_14', 'roc_3', 'roc_7',
    'bb_pb',
    'news_sentiment',
    'ofi_z',                # order flow imbalance Z-score
    'vwap_dist',            # distance from VWAP
    'liq_proximity',        # proximity to liquidation zone
    # strategy-conditioned features (meta-learning signal filter)
    'trend_strength',       # Regime: EMA spread / ATR
    'vol_regime',           # Regime: short-term vol vs long-term vol
    'is_trending',          # Regime: binary flag
    'is_volatile',          # Regime: binary flag
    'signal_rsi', 'signal_macd', 'signal_bb',
    # Market-context features (exact join, no resampling):
    'oi_change_pct',    # 1h OI % change (Binance); 0 for non-1h TFs
    'fear_greed_norm',  # Fear & Greed 0-1 scaled (daily); 0.5 for non-1d TFs
    'liq_long_z',       # rolling z-score of long liquidations USD (Coinglass); 0 if no key
    'liq_short_z',      # rolling z-score of short liquidations USD; 0 if no key
    'liq_dom',          # short_liq / total_liq — squeeze bias (0.5 neutral); 0 if no key
    # CoinGlass v4 features (0.0 when data not downloaded yet — stable schema)
    'oi_close',         # open interest USD close
    'ls_ratio',         # global long/short account ratio
    'ls_long_pct',      # % accounts long
    'ls_short_pct',     # % accounts short
    'fr_close',         # funding rate close
    'liq_long_usd',     # long liquidations USD
    'liq_short_usd',    # short liquidations USD
    'fut_taker_buy_usd',    # futures taker buy volume
    'fut_taker_sell_usd',   # futures taker sell volume
    'taker_cvd',        # cumulative volume delta (futures)
    'cbp_premium_rate', # Coinbase premium index
    'fear_greed',       # Fear & Greed raw 0-100
    'btc_dominance',    # BTC market dominance %
    'stablecoin_mcap',  # stablecoin market cap proxy
    'symbol_id',        # ordinal symbol encoding (1-based; 0 = unknown)
    # Explicit regime one-hot features
    'regime_bull', 'regime_bear', 'regime_chop', 'regime_high_vol',
]


def prepare_data(filepath, timeframe: str = '1h', symbol: str | None = None):
    if symbol:
        log.info("Loading data for %s/%s from parquet...", symbol, timeframe)
        df = load_ohlcv(symbol, timeframe)
        if df.empty:
            raise FileNotFoundError(f"No OHLCV data for {symbol}/{timeframe}")
    else:
        log.info("Loading data from %s...", filepath)
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Phase 4 rollout — F1 data-integrity gate.
    try:
        from src.utils.data_quality import validate_ohlcv, DataQualityError
        df, _dq = validate_ohlcv(df, symbol=symbol or '', timeframe=timeframe)
        if _dq.soft_warnings:
            log.info("[base][%s/%s] data quality: %s",
                     symbol, timeframe, _dq.soft_warnings[:3])
    except Exception as e:
        from src.utils.data_quality import DataQualityError
        if isinstance(e, DataQualityError):
            raise
        log.warning("[base][%s/%s] data_quality check skipped: %s",
                    symbol, timeframe, e)

    log.info("Engineering features...")
    # Fractional differencing replaces pct_change
    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()  # keep for legacy lag features

    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['volatility'] = df['return'].rolling(window=7).std()
    df['dist_sma_7'] = df['close'] / df['sma_7'] - 1
    df['dist_sma_30'] = df['close'] / df['sma_30'] - 1

    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bollinger_bands(df, window=20)
    df = add_roc(df, [3, 7, 14])
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)
    df = add_atr(df, 14)
    df = add_ofi(df)
    df = add_vwap(df)
    df = add_liquidity_proximity(df)

    # Regime Features
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['trend_strength'] = (df['ema_fast'] - df['ema_slow']).abs() / df['atr_14'].replace(0, 1e-9)
    df['vol_long'] = df['return'].rolling(window=100, min_periods=10).std()
    df['vol_regime'] = df['volatility'] / df['vol_long'].replace(0, 1e-9)
    df['is_trending'] = (df['trend_strength'] > 1.5).astype(int)
    df['is_volatile'] = (df['vol_regime'] > 1.5).astype(int)


    df['vol_sma_14'] = df['volume'].rolling(window=14).mean()
    df['volume_momentum'] = df['volume'] / df['vol_sma_14']

    df['high_14'] = df['high'].rolling(window=14).max()
    df['low_14'] = df['low'].rolling(window=14).min()
    hl_diff = (df['high_14'] - df['low_14']).replace(0, 0.0001)
    df['stoch_k'] = (df['close'] - df['low_14']) / hl_diff * 100

    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['return_lag3'] = df['return'].shift(3)
    df['return_lag5'] = df['return'].shift(5)
    df['atr_pct'] = (df['high'] - df['low']) / df['close']

    # Strategy-conditioned features — model learns when each signal is reliable
    df['signal_rsi'] = 0.0
    df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1.0
    df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1.0
    df['signal_macd'] = np.where(df['macd_hist'] > 0, 1.0, -1.0)
    df['signal_bb'] = 0.0
    df.loc[df['bb_pb'] < 0.1, 'signal_bb'] = 1.0
    df.loc[df['bb_pb'] > 0.9, 'signal_bb'] = -1.0

    news_path = os.path.join(base_dir, 'data', 'raw', 'cryptocompare_news.csv')
    if not os.path.exists(news_path):
        log.warning(
            "News sentiment CSV not found: %s -- 'news_sentiment' feature will be 0.0 for all rows",
            news_path,
        )
    df = add_news_sentiment(df, news_path)

    # ── Market-context features (exact join, no resampling) ───────────────
    # 1h/4h/1d/1w: Open Interest % change (Binance Futures 1h data).
    # 4h/1d/1w bar opens always land on 1h boundaries -> exact timestamp match, no resampling.
    # 15m/5m bars don't align -> skip (intermediate timestamps have no OI entry).
    if timeframe in ('1h', '4h', '1d', '1w') and symbol:
        try:
            oi = load_open_interest(symbol)
            if not oi.empty:
                oi = oi[["timestamp", "oi_usdt"]].copy()
                oi["oi_change_pct"] = oi["oi_usdt"].pct_change().clip(-1, 1)
                df = df.merge(oi[["timestamp", "oi_change_pct"]], on="timestamp", how="left")
                df["oi_change_pct"] = df["oi_change_pct"].fillna(0.0)
                oi_matched = df["oi_change_pct"].ne(0).sum()
                log.info("[%s/%s] OI join: %d/%d bars matched",
                         symbol, timeframe, oi_matched, len(df))
            else:
                df["oi_change_pct"] = 0.0
        except Exception as e:
            log.warning("OI join failed for %s: %s", symbol, e)
            df["oi_change_pct"] = 0.0
    else:
        df["oi_change_pct"] = 0.0

    # 1h only: Liquidation data (Coinglass — requires COINGLASS_API_KEY).
    # liq_long_z / liq_short_z: rolling 24h z-score of USD liquidated.
    # liq_dom: short_liq / total_liq — high = more shorts squeezed = bullish bias.
    # Falls back to 0 silently when no key or no data.
    if timeframe == '1h' and symbol:
        try:
            liq = load_liquidations(symbol)
            if not liq.empty:
                liq = liq[["timestamp", "liq_long_usd", "liq_short_usd", "liq_total_usd"]].copy()
                _w = 24  # 24-bar rolling window for z-score
                for col, zcol in [("liq_long_usd", "liq_long_z"), ("liq_short_usd", "liq_short_z")]:
                    mu = liq[col].rolling(_w, min_periods=1).mean()
                    sd = liq[col].rolling(_w, min_periods=1).std().replace(0, 1e-9)
                    liq[zcol] = ((liq[col] - mu) / sd).clip(-4, 4)
                liq["liq_dom"] = (liq["liq_short_usd"] / (liq["liq_total_usd"] + 1e-9)).clip(0, 1)
                df = df.merge(
                    liq[["timestamp", "liq_long_z", "liq_short_z", "liq_dom"]],
                    on="timestamp", how="left",
                )
                for col in ("liq_long_z", "liq_short_z", "liq_dom"):
                    df[col] = df[col].fillna(0.0)
                matched = df["liq_long_z"].ne(0).sum()
                log.info("[%s/1h] Liq join: %d/%d bars matched", symbol, matched, len(df))
            else:
                df["liq_long_z"] = 0.0
                df["liq_short_z"] = 0.0
                df["liq_dom"]     = 0.0
        except Exception as e:
            log.warning("Liquidation join failed for %s: %s", symbol, e)
            df["liq_long_z"] = 0.0
            df["liq_short_z"] = 0.0
            df["liq_dom"]     = 0.0
    else:
        df["liq_long_z"] = 0.0
        df["liq_short_z"] = 0.0
        df["liq_dom"]     = 0.0

    # 1d only: Fear & Greed index (normalized 0-1, 0.5 = neutral)
    if timeframe == '1d':
        try:
            fg = load_fear_greed()
            if not fg.empty:
                fg = fg[["timestamp", "fear_greed"]].copy()
                fg["fear_greed_norm"] = fg["fear_greed"] / 100.0
                # date-only join: strip time component for exact match
                df["_date"] = df["timestamp"].dt.normalize()
                fg["_date"] = fg["timestamp"].dt.normalize()
                df = df.merge(fg[["_date", "fear_greed_norm"]], on="_date", how="left")
                df = df.drop(columns=["_date"])
                df["fear_greed_norm"] = df["fear_greed_norm"].fillna(0.5)
                log.info("[%s/1d] FG join: %d rows matched",
                         symbol, df["fear_greed_norm"].ne(0.5).sum())
            else:
                df["fear_greed_norm"] = 0.5
        except Exception as e:
            log.warning("Fear & Greed join failed: %s", e)
            df["fear_greed_norm"] = 0.5
    else:
        df["fear_greed_norm"] = 0.5

    # CoinGlass v4 enrichment — OI, L/S ratio, funding, liquidations, taker
    # flow, Coinbase premium, and macro (fear/greed, BTC dominance, stablecoin
    # mcap). Stable schema: all columns default to 0.0 if files not present.
    if symbol:
        try:
            df = add_coinglass_features(df, symbol, timeframe)
            log.info("[base][%s/%s] CoinGlass features merged", symbol, timeframe)
        except Exception as _cg_exc:
            log.warning("[base][%s/%s] CoinGlass features skipped: %s",
                        symbol, timeframe, _cg_exc)

    # Dynamic Volatility-based Triple Barrier (asymmetric: pt=4, sl=2)
    # Wizard 2026-05-16: symmetric 2:2 collapsed AUC to ~0.50 (noise floor).
    # 2:1 ratio = let winners run, cut losers tight — matches trend trainer.
    labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=4.0, sl_multiplier=2.0, max_bars=24)
    df['target_raw'] = labels
    df['t1_timestamp'] = t1_times
    
    # Remove timeouts (0) to formulate a strict binary classification: TP hit vs SL hit
    df = df[df['target_raw'] != 0].copy()
    df['target_tb'] = (df['target_raw'] == 1).astype(int)

    df = df.dropna()
    stats = label_stats(df['target_raw'])
    log.info("Triple Barrier label distribution: %s", stats)
    return df


def train_model(timeframe: str = '1h'):
    """Train the base directional classifier at a given timeframe.

    2026-05-15 — wall-clock instrumentation: records started_at, finished_at,
    duration_s in the meta JSON + emits a "single-epoch" record to the
    live training-progress tracker so the dashboard's epoch column shows
    base/trend/futures/scalping/meta runs alongside TFT's per-epoch view.

    timeframe — one of 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1mo. Drives both the
    input file (data/raw/<sym>_<tf>.csv.gz) and the output artifact name
    (per src.utils.model_paths). Default 1h matches the canonical (legacy)
    behaviour — when called at 1h the trainer ALSO writes the legacy
    btc_rf_model.joblib so the bot's inference path stays unchanged.

    CIO overrides: any operator-approved values in
    `models.base.cio_overrides` (set via CIOAgent.apply_best) are merged
    into the HP block by _load_model_params() (schema-bounded), and also
    recorded in the meta JSON for audit.
    """
    # 2026-05-15 — wall-clock instrumentation. Captures start unix-time so
    # the meta JSON can carry duration_s after the trainer completes.
    import time as _time
    _train_started_at = _time.time()
    _train_task_id = f"base_{timeframe}_{int(_train_started_at)}"
    try:
        from src.utils import training_progress as _tp
        _tp.start(_train_task_id, model="base", tf=timeframe,
                  n_epochs=1, trainer="train_model.py")
    except Exception as _e:
        log.warning("[base] progress.start failed: %s", _e)

    from src.utils.cio_overrides import load_cio_overrides
    cio = load_cio_overrides('base')
    if cio:
        log.info("[CIO overrides] base/%s read: %s", timeframe, cio)

    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT', 'ETH_USDT']
    _syms_sorted = sorted(symbols)
    all_data = []
    for sym in symbols:
        log.info("Processing %s...", sym)
        try:
            df = prepare_data(None, timeframe=timeframe, symbol=sym)
            df = add_explicit_regime(df)
            df['symbol_id'] = float(_syms_sorted.index(sym) + 1)
            all_data.append(df)
        except FileNotFoundError:
            log.warning("Data for %s not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6 * 365)
            try:
                df = prepare_data(None, timeframe=timeframe, symbol=sym)
                df = add_explicit_regime(df)
                df['symbol_id'] = float(_syms_sorted.index(sym) + 1)
                all_data.append(df)
            except Exception as e:
                log.error("Failed to prepare %s after download: %s", sym, e)
        except Exception as e:
            log.error("Failed to prepare %s: %s", sym, e)

    if not all_data:
        log.error("No data found to train the model.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    combined_df.set_index('timestamp', inplace=True)

    missing = [f for f in FEATURE_COLUMNS if f not in combined_df.columns]
    if missing:
        log.warning("Missing features (will fill with 0): %s", missing)
        for col in missing:
            combined_df[col] = 0.0

    X = combined_df[FEATURE_COLUMNS].fillna(0)
    y = combined_df['target_tb']

    log.info("Dataset: %d total samples | %d features | symbols: %s",
             len(combined_df), len(FEATURE_COLUMNS), symbols)

    # B3: load HP from training_rules.json (guards: missing-key warn, type/range
    # validate, schema check, version+hash saved to meta for audit trail).
    # training_rules.json uses max_depth=8; the old hard-coded value was 6.
    hp, rules_version, params_hash = _load_model_params('base')

    # Walk-forward cross-validation
    t1_series = combined_df['t1_timestamp']
    _close_returns = combined_df['close'].pct_change().fillna(0)
    cv = PurgedKFold(n_splits=5, t1=t1_series, embargo_td=pd.Timedelta(hours=48))
    fold_accuracies = []
    in_sample_fold_accs = []  # P3: in-sample accuracy per fold for overfit_ratio
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        base_clf = make_classifier(
            random_state=42,
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=0.03,
            l2_regularization=0.5,
            early_stopping=True,
            class_weight=hp['class_weight'],
        )
        weights = compute_afml_weights(y.iloc[train_idx], t1_series.iloc[train_idx], _close_returns.iloc[train_idx])
        base_clf.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights)
        fold_acc = accuracy_score(y.iloc[test_idx], base_clf.predict(X.iloc[test_idx]))
        fold_accuracies.append(fold_acc)
        in_sample_fold_accs.append(
            accuracy_score(y.iloc[train_idx], base_clf.predict(X.iloc[train_idx]))
        )
        log.info("Walk-forward fold %d/%d: accuracy=%.2f%%", fold_i + 1, cv.n_splits, fold_acc * 100)

    log.info("Walk-forward mean accuracy: %.2f%% +/- %.2f%%",
             np.mean(fold_accuracies) * 100, np.std(fold_accuracies) * 100)

    # P3: overfit ratio
    _wf_mean_acc = float(np.mean(fold_accuracies))
    _in_sample_mean_acc = float(np.mean(in_sample_fold_accs)) if in_sample_fold_accs else None
    _overfit_ratio: float | None = None
    if _in_sample_mean_acc is not None and _in_sample_mean_acc > 0:
        _overfit_ratio = (_in_sample_mean_acc - _wf_mean_acc) / _in_sample_mean_acc
        if _overfit_ratio > 0.20:
            log.error("[base] overfit_ratio=%.3f > 0.20 (in_sample=%.2f%% wf=%.2f%%) -- model is memorising",
                      _overfit_ratio, _in_sample_mean_acc * 100, _wf_mean_acc * 100)
        elif _overfit_ratio > 0.10:
            log.warning("[base] overfit_ratio=%.3f > 0.10 (in_sample=%.2f%% wf=%.2f%%)",
                        _overfit_ratio, _in_sample_mean_acc * 100, _wf_mean_acc * 100)

    hard_gate_wf(_wf_mean_acc, 'base')

    # Final model — B2: three non-overlapping splits
    #   train  [0 : 70%]   — model fit
    #   cal    [70% : 85%] — isotonic calibration (held out, no leakage)
    #   test   [85% : 100%]— evaluation only
    log.info("Training final model with isotonic probability calibration (3-split)...")
    n = len(X)
    train_end = int(n * 0.70)
    cal_end   = int(n * 0.85)
    base_clf = make_classifier(
        random_state=42,
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=0.03,
        l2_regularization=0.5,
        early_stopping=True,
        class_weight=hp['class_weight'],
    )
    calib_start_time = combined_df.index[train_end]
    valid_train_mask = combined_df['t1_timestamp'].iloc[:train_end] < calib_start_time
    safe_train_idx = np.arange(train_end)[valid_train_mask]

    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    weights_calib = compute_afml_weights(y.iloc[safe_train_idx], t1_series.iloc[safe_train_idx], _close_returns.iloc[safe_train_idx])
    base_clf.fit(X.iloc[safe_train_idx], y.iloc[safe_train_idx], sample_weight=weights_calib)
    calibrated.fit(X.iloc[train_end:cal_end], y.iloc[train_end:cal_end])
    _best_thr, _best_thr_score = find_optimal_threshold(
        calibrated, X.iloc[train_end:cal_end], y.iloc[train_end:cal_end],
        _close_returns.iloc[train_end:cal_end],
    )

    X_test = X.iloc[cal_end:]
    y_test = y.iloc[cal_end:]
    predictions = calibrated.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 500)
    # PR-44: AUC + win-precision so the dashboard's AUC / Win Prec%
    # columns aren't blank for everything except the meta-labeler.
    try:
        proba_test = calibrated.predict_proba(X_test)[:, 1]
    except Exception:
        proba_test = None

    log.info("Base Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    # ── Persist artifacts via the canonical model_paths helper ───────────
    # Always writes models/<key>_<tf>_model.joblib + <key>_<tf>_meta.json.
    # When tf == CANONICAL_TF[key] (1h for base), ALSO writes the legacy
    # btc_rf_model.joblib + btc_rf_model_meta.json so the bot's inference
    # engine (which still loads the legacy paths) stays compatible.
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    from src.utils.model_integrity import sign_model
    paths = artifact_paths('base', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    sign_model(str(paths['model']))
    log.info("Model saved -> %s", paths['model'])

    # Phase 6b wire-in — persist training feature distribution baseline.
    try:
        from src.risk.drift_baseline import save_baseline
        save_baseline('base', timeframe, X)
    except Exception as _e:
        log.warning("[base][%s] save_baseline failed: %s", timeframe, _e)

    # 2026-05-15 — capture finished_at + duration before writing meta so
    # downstream consumers (training_history, dashboard) have the actual
    # wall-clock cost of this run.
    _train_finished_at = _time.time()
    _train_duration_s = _train_finished_at - _train_started_at
    meta = {
        "model": "Base (HistGBT + Calibrated)",
        "accuracy": accuracy * 100,
        "optimal_threshold": _best_thr,
        "optimal_sortino": round(_best_thr_score, 4),
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": train_end, "n_test": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),  # required by MLPredictor._get_model_features
        "cio_overrides_applied": dict(cio) if cio else None,
        "n_iterations": n_iter,
        "walk_forward_mean_acc": round(float(np.mean(fold_accuracies)) * 100, 2),
        "walk_forward_std_acc": round(float(np.std(fold_accuracies)) * 100, 2),
        "wf_fold_scores": [round(v, 6) for v in fold_accuracies],
        "in_sample_mean_acc": round(_in_sample_mean_acc * 100, 2) if _in_sample_mean_acc else None,
        "overfit_ratio": round(_overfit_ratio, 6) if _overfit_ratio is not None else None,
        "target": "triple_barrier_long_win",
        "symbols": symbols, "symbols_sorted": _syms_sorted, "timeframe": timeframe,
        "last_trained": datetime.now(timezone.utc).isoformat(),
        # 2026-05-15 — wall-clock instrumentation.
        "started_at_unix":  _train_started_at,
        "finished_at_unix": _train_finished_at,
        "duration_s":       round(_train_duration_s, 2),
        "epochs_completed": 1,
        "per_epoch_s":      round(_train_duration_s, 2),
        # B3: audit trail — which rules version + HP hash produced this artifact
        "hp_rules_version": rules_version,
        "hp_params_hash": params_hash,
        "hp": hp,
    }
    try:
        _tp.epoch_done(_train_task_id, 1, _train_duration_s)
        _tp.finish(_train_task_id, status="done")
    except Exception as _e:
        log.warning("[base] progress.finish failed: %s", _e)
    if proba_test is not None:
        from src.utils.model_metrics import merge_metrics_into_meta
        merge_metrics_into_meta(meta, y_test, proba_test)
    write_json(str(paths['meta']), meta)
    try:
        from src.engine.champion_challenger import ChampionRegistry
        meta['model_path'] = str(paths['model'])
        ChampionRegistry().register_challenger('base', timeframe, meta)
    except Exception as _cc_e:
        log.warning("[base] champion_challenger failed: %s", _cc_e)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        sign_model(str(paths['legacy_model']))
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)

    # Phase K (2026-05-14) — record run in training history.
    try:
        from src.analytics.training_history import record_run_from_meta
        record_run_from_meta(meta, model='base', tf=timeframe,
                             trainer='train_model.py',
                             meta_path=str(paths['meta']))
    except Exception as e:
        log.warning("[base] record_run skipped: %s", e)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the base directional model")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_model(timeframe=args.timeframe)
