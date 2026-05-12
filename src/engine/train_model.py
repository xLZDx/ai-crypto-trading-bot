import os
import sys
import json
import logging
import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_base')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features,
    add_ofi, add_vwap, add_funding_zscore, add_liquidity_proximity, add_atr,
    add_news_sentiment
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats


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
]


def prepare_data(filepath):
    log.info("Loading data from %s...", filepath)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

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
    df = add_news_sentiment(df, news_path)

    # Dynamic Volatility-based Triple Barrier
    labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=2.0, sl_multiplier=2.0, max_bars=24)
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

    timeframe — one of 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1mo. Drives both the
    input file (data/raw/<sym>_<tf>.csv.gz) and the output artifact name
    (per src.utils.model_paths). Default 1h matches the canonical (legacy)
    behaviour — when called at 1h the trainer ALSO writes the legacy
    btc_rf_model.joblib so the bot's inference path stays unchanged.
    """
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT', 'ETH_USDT']
    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_{timeframe}.csv.gz')
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_{timeframe}.csv.gz')
        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
            log.info("  Using archive data for %s: %s", sym, archive_path)
        if not os.path.exists(full_data_path):
            log.warning("Data for %s not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6 * 365)

        if os.path.exists(full_data_path):
            log.info("Processing %s...", sym)
            try:
                df = prepare_data(full_data_path)
                all_data.append(df)
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

    # Walk-forward cross-validation
    t1_series = combined_df['t1_timestamp']
    # Embargo = 2 * horizon (24 bars for base model)
    pct_embargo = (2.0 * 24) / len(X)
    cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=pct_embargo)
    fold_accuracies = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        base_clf = make_classifier(
            random_state=42, n_estimators=500, max_depth=6,
            learning_rate=0.03, l2_regularization=0.5,
            early_stopping=True, class_weight='balanced'
        )
        weights = compute_sample_weight('balanced', y.iloc[train_idx])
        base_clf.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights)
        fold_acc = accuracy_score(y.iloc[test_idx], base_clf.predict(X.iloc[test_idx]))
        fold_accuracies.append(fold_acc)
        log.info("Walk-forward fold %d/%d: accuracy=%.2f%%", fold_i + 1, cv.n_splits, fold_acc * 100)

    log.info("Walk-forward mean accuracy: %.2f%% ± %.2f%%",
             np.mean(fold_accuracies) * 100, np.std(fold_accuracies) * 100)

    # Final model on full data with probability calibration
    log.info("Training final model with isotonic probability calibration...")
    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = make_classifier(
        random_state=42, n_estimators=500, max_depth=6,
        learning_rate=0.03, l2_regularization=0.5,
        early_stopping=True, class_weight='balanced'
    )
    calib_start_time = combined_df.index[calib_split]
    valid_train_mask = combined_df['t1_timestamp'].iloc[:calib_split] < calib_start_time
    safe_train_idx = np.arange(calib_split)[valid_train_mask]
    
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    weights_calib = compute_sample_weight('balanced', y.iloc[safe_train_idx])
    base_clf.fit(X.iloc[safe_train_idx], y.iloc[safe_train_idx], sample_weight=weights_calib)
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])

    X_test = X.iloc[int(n * 0.90):]
    y_test = y.iloc[int(n * 0.90):]
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

    meta = {
        "model": "Base (HistGBT + Calibrated)",
        "accuracy": accuracy * 100,
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
        "n_features": len(FEATURE_COLUMNS), "n_iterations": n_iter,
        "walk_forward_mean_acc": round(float(np.mean(fold_accuracies)) * 100, 2),
        "walk_forward_std_acc": round(float(np.std(fold_accuracies)) * 100, 2),
        "target": "triple_barrier_long_win",
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the base directional model")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_model(timeframe=args.timeframe)
