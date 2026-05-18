import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from src.data_ingestion.ohlcv_parquet_loader import load_ohlcv
from sklearn.ensemble import HistGradientBoostingClassifier  # kept for type compat
from sklearn.calibration import CalibratedClassifierCV
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from src.utils.purged_kfold import PurgedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_futures')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_rsi, add_roc, add_time_features, add_atr,
    add_ofi, add_funding_zscore, add_liquidity_proximity, add_coinglass_features,
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats


FEATURE_COLUMNS = [
    'frac_diff_d40',
    'rsi_14',
    'dist_to_support',
    'volume_drop',
    'hour',
    'roc_5',
    'ofi_z',
    'funding_z',
    'funding_positive',  # binary: funding > 0.1% → shorts paid
    'funding_negative',  # binary: funding < -0.05% → longs paid
    'dist_to_supply',
    'liq_proximity',
    'signal_rsi',        # strategy signal as feature
    'trend_strength',
    'vol_regime',
    'is_trending',
    'is_volatile',
    'news_sentiment',
    # CoinGlass v4 features (0.0 when data not downloaded yet — stable schema)
    'oi_close', 'ls_ratio', 'ls_long_pct', 'ls_short_pct',
    'fr_close', 'liq_long_usd', 'liq_short_usd',
    'fut_taker_buy_usd', 'fut_taker_sell_usd', 'taker_cvd',
    'cbp_premium_rate', 'fear_greed', 'btc_dominance', 'stablecoin_mcap',
]


def prepare_futures_data(filepath, timeframe: str = '1h', symbol: str | None = None):
    if symbol:
        log.info("Loading data for Futures Pipeline: %s/%s from parquet...", symbol, timeframe)
        df = load_ohlcv(symbol, timeframe)
        if df.empty:
            raise FileNotFoundError(f"No OHLCV data for {symbol}/{timeframe}")
    else:
        log.info("Loading data for Futures (Shorting) Pipeline from %s...", filepath)
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Phase 4 rollout — F1 data-integrity gate (schema + bounds + freshness).
    try:
        from src.utils.data_quality import validate_ohlcv, DataQualityError
        df, _dq = validate_ohlcv(df, symbol=symbol or '', timeframe=timeframe)
        if _dq.soft_warnings:
            log.info("[futures][%s/%s] data quality: %s",
                     symbol, timeframe, _dq.soft_warnings[:3])
    except Exception as e:
        from src.utils.data_quality import DataQualityError
        if isinstance(e, DataQualityError):
            raise
        log.warning("[futures][%s/%s] data_quality check skipped: %s",
                    symbol, timeframe, e)

    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()
    df = add_rsi(df, 14)
    df = add_roc(df, [5])
    df = add_time_features(df)
    df = add_atr(df, 14)
    df = add_ofi(df)
    df = add_funding_zscore(df)
    df = add_liquidity_proximity(df)

    # Regime Features
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['trend_strength'] = (df['ema_fast'] - df['ema_slow']).abs() / df['atr_14'].replace(0, 1e-9)
    df['vol_short'] = df['return'].rolling(window=7).std()
    df['vol_long'] = df['return'].rolling(window=100, min_periods=10).std()
    df['vol_regime'] = df['vol_short'] / df['vol_long'].replace(0, 1e-9)
    df['is_trending'] = (df['trend_strength'] > 1.5).astype(int)
    df['is_volatile'] = (df['vol_regime'] > 1.5).astype(int)

    df['low_30'] = df['low'].rolling(30).min()
    df['dist_to_support'] = (df['close'] - df['low_30']) / df['close']

    df['vol_sma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_drop'] = (df['volume'] < df['vol_sma_7'] * 0.7).astype(int)

    # Strategy signal as feature
    df['signal_rsi'] = 0.0
    df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1.0
    df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1.0

    # Phase I (2026-05-14) — wire news sentiment (parquet → CSV fallback).
    try:
        from src.analysis.feature_engineering import add_news_sentiment
        _news_csv = os.path.join(base_dir, 'data', 'raw', 'cryptocompare_news.csv')
        df = add_news_sentiment(df, _news_csv)
    except Exception as e:
        log.warning("[futures] add_news_sentiment skipped: %s", e)
        if 'news_sentiment' not in df.columns:
            df['news_sentiment'] = 0.0

    if symbol:
        try:
            df = add_coinglass_features(df, symbol, timeframe)
            log.info("[futures][%s/%s] CoinGlass features merged", symbol, timeframe)
        except Exception as _cg_exc:
            log.warning("[futures][%s/%s] CoinGlass features skipped: %s",
                        symbol, timeframe, _cg_exc)

    # Triple barrier for SHORTS: dynamic ATR-based barriers (asymmetric pt=4, sl=2)
    # Wizard 2026-05-16: symmetric 2:2 produced AUC noise floor on futures.
    # For SHORTS: pt=4*ATR profit target (price drop), sl=2*ATR loss cap (price rise).
    labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=4.0, sl_multiplier=2.0, max_bars=12)
    df['target_raw'] = labels
    df['t1_timestamp'] = t1_times
    
    # Remove timeouts
    df = df[df['target_raw'] != 0].copy()
    
    # For futures short model: label 1 = "short win" = price fell (barrier -1 hit first)
    df['target_short'] = (df['target_raw'] == -1).astype(int)
    df = df.dropna()
    log.info("Futures TB distribution: %s", label_stats(labels))
    return df


def train_futures_model(timeframe: str = '1h'):
    """Train the futures-short classifier at a given timeframe.

    CIO overrides from `models.futures.cio_overrides` are logged + recorded
    in meta JSON. Per-HP merging is deferred (see Sprint 1A R1).
    """
    from src.utils.cio_overrides import load_cio_overrides
    cio = load_cio_overrides('futures')
    if cio:
        log.info("[CIO overrides] futures/%s: %s (NOT auto-merged into params yet)",
                 timeframe, cio)

    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']

    all_data = []
    for sym in symbols:
        log.info("Processing %s...", sym)
        try:
            df = prepare_futures_data(None, timeframe=timeframe, symbol=sym)
            all_data.append(df)
        except FileNotFoundError:
            log.warning("Data for %s not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6 * 365)
            try:
                df = prepare_futures_data(None, timeframe=timeframe, symbol=sym)
                all_data.append(df)
            except Exception as e:
                log.error("Failed to prepare %s after download: %s", sym, e)
        except Exception as e:
            log.error("Failed to prepare %s: %s", sym, e)

    if not all_data:
        log.error("No data found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    combined_df.set_index('timestamp', inplace=True)

    for col in [f for f in FEATURE_COLUMNS if f not in combined_df.columns]:
        combined_df[col] = 0.0

    X = combined_df[FEATURE_COLUMNS].fillna(0)
    y = combined_df['target_short']

    log.info("Futures dataset: %d total | features %d | symbols %s | timeframe %s",
             len(combined_df), len(FEATURE_COLUMNS), symbols, timeframe)

    # ── CIO overrides MERGE (X1.3, 2026-05-13) — schema-bounded ────────────
    from src.utils.cio_overrides import merge_with_defaults as _merge
    _FUT_HP_DEFAULTS = {
        'n_estimators': 400, 'max_depth': 6,
        'learning_rate': 0.03, 'l2_regularization': 0.2,
        'class_weight': 'balanced',
    }
    _FUT_HP_SCHEMA = {
        'n_estimators':      (int,   1,    10_000),
        'max_depth':         (int,   1,    50),
        'learning_rate':     (float, 1e-4, 1.0),
        'l2_regularization': (float, 0.0,  10.0),    # reviewer-tightened from 100
        'class_weight':      (str,   None, None),
    }
    _fut_hp, _fut_applied = _merge('futures', _FUT_HP_DEFAULTS, _FUT_HP_SCHEMA)

    t1_series = combined_df['t1_timestamp']
    # Embargo = 2 * horizon (12 bars for futures model)
    pct_embargo = (2.0 * 12) / len(X)
    cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=pct_embargo)
    fold_accs = []
    in_sample_fold_accs = []  # P3
    for i, (tr, te) in enumerate(cv.split(X)):
        clf = make_classifier(
            random_state=42,
            n_estimators=_fut_hp['n_estimators'],
            max_depth=_fut_hp['max_depth'],
            learning_rate=_fut_hp['learning_rate'],
            l2_regularization=_fut_hp['l2_regularization'],
            class_weight=_fut_hp['class_weight'],
            early_stopping=True,
        )
        weights = compute_sample_weight('balanced', y.iloc[tr])
        clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights)
        fold_accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
        in_sample_fold_accs.append(accuracy_score(y.iloc[tr], clf.predict(X.iloc[tr])))
        log.info("Futures walk-forward fold %d: %.2f%%", i + 1, fold_accs[-1] * 100)

    log.info("Futures walk-forward mean: %.2f%% +/- %.2f%%",
             np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    # P3: overfit ratio
    _wf_mean = float(np.mean(fold_accs))
    _in_sample_mean = float(np.mean(in_sample_fold_accs)) if in_sample_fold_accs else None
    _overfit_ratio: float | None = None
    if _in_sample_mean is not None and _in_sample_mean > 0:
        _overfit_ratio = (_in_sample_mean - _wf_mean) / _in_sample_mean
        if _overfit_ratio > 0.20:
            log.error("[futures] overfit_ratio=%.3f > 0.20 (in_sample=%.2f%% wf=%.2f%%) -- model is memorising",
                      _overfit_ratio, _in_sample_mean * 100, _wf_mean * 100)
        elif _overfit_ratio > 0.10:
            log.warning("[futures] overfit_ratio=%.3f > 0.10 (in_sample=%.2f%% wf=%.2f%%)",
                        _overfit_ratio, _in_sample_mean * 100, _wf_mean * 100)

    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = make_classifier(
        random_state=42,
        n_estimators=_fut_hp['n_estimators'],
        max_depth=_fut_hp['max_depth'],
        learning_rate=_fut_hp['learning_rate'],
        l2_regularization=_fut_hp['l2_regularization'],
        class_weight=_fut_hp['class_weight'],
        early_stopping=True,
    )
    calib_start_time = combined_df.index[calib_split]
    valid_train_mask = combined_df['t1_timestamp'].iloc[:calib_split] < calib_start_time
    safe_train_idx = np.arange(calib_split)[valid_train_mask]
    
    weights_calib = compute_sample_weight('balanced', y.iloc[safe_train_idx])
    base_clf.fit(X.iloc[safe_train_idx], y.iloc[safe_train_idx], sample_weight=weights_calib)
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])

    X_test = X.iloc[int(n * 0.90):]
    y_test = y.iloc[int(n * 0.90):]
    predictions = calibrated.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    short_acc = report.get('1', {}).get('precision', 0.0) * 100
    long_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 400)
    # PR-44 — populate AUC + win-precision so the dashboard column is filled.
    try:
        proba_test = calibrated.predict_proba(X_test)[:, 1]
    except Exception:
        proba_test = None

    log.info("Futures Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    # ── Persist via canonical model_paths helper ──────────────────────────
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    from src.utils.model_integrity import sign_model
    paths = artifact_paths('futures', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    sign_model(str(paths['model']))
    log.info("Model saved -> %s", paths['model'])

    # Phase 6b wire-in — persist training feature distribution baseline.
    try:
        from src.risk.drift_baseline import save_baseline
        save_baseline('futures', timeframe, X)
    except Exception as _e:
        log.warning("[futures][%s] save_baseline failed: %s", timeframe, _e)

    meta = {
        "model": "Futures Short (HistGBT + Calibrated)",
        "accuracy": accuracy * 100,
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),  # required by MLPredictor._get_model_features
        "cio_overrides_applied": dict(_fut_applied) if _fut_applied else None,  # X1.3
        "n_iterations": n_iter,
        "walk_forward_mean_acc": round(float(np.mean(fold_accs)) * 100, 2),
        "wf_fold_scores": [round(v, 6) for v in fold_accs],
        "in_sample_mean_acc": round(_in_sample_mean * 100, 2) if _in_sample_mean else None,
        "overfit_ratio": round(_overfit_ratio, 6) if _overfit_ratio is not None else None,
        "target": "triple_barrier_short_win",
        "symbols": symbols, "timeframe": timeframe,
        "last_trained": datetime.now(timezone.utc).isoformat()
    }
    if proba_test is not None:
        from src.utils.model_metrics import merge_metrics_into_meta
        merge_metrics_into_meta(meta, y_test, proba_test)
    write_json(str(paths['meta']), meta)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        sign_model(str(paths['legacy_model']))
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)

    # Phase K (2026-05-14) — record run in training history.
    try:
        from src.analytics.training_history import record_run_from_meta
        record_run_from_meta(meta, model='futures', tf=timeframe,
                             trainer='train_futures_model.py',
                             meta_path=str(paths['meta']))
    except Exception as e:
        log.warning("[futures] record_run skipped: %s", e)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the futures-short model")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_futures_model(timeframe=args.timeframe)
