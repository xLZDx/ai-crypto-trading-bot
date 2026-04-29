import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_scalping')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features, resample_1s_to_1m,
    add_ofi, add_vwap, add_keltner
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats


FEATURE_COLUMNS = [
    'frac_diff_d40',
    'rsi_7',
    'macd_fast',
    'volume_surge',
    'dist_to_micro_supp',
    'taker_buy_ratio', 'avg_trade_size',
    'hour',
    'roc_3', 'roc_5', 'roc_10',
    'bb_pb',
    'ofi_z',           # OFI Z-score — key for 1m microstructure
    'vwap_dist',       # VWAP distance at 1m granularity
    'kc_pos',          # Keltner position (volatility breakout)
    'signal_rsi', 'signal_bb',  # strategy signals as features
]


def _walk_forward_splits(n, n_splits=5, test_pct=0.10, gap_pct=0.05):
    gap = int(n * gap_pct)
    test_size = int(n * test_pct)
    train_end_start = int(n * 0.50)
    step = max((n - train_end_start - test_size - gap) // max(n_splits - 1, 1), 1)
    splits = []
    for i in range(n_splits):
        train_end = train_end_start + i * step
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n:
            break
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def _engineer_scalping_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()
    df = add_rsi(df, period=7, col_name='rsi_7')
    df = add_macd(df, fast=5, slow=13, signal=3, prefix='')
    df.rename(columns={
        'macd': 'macd_fast', 'macd_signal': 'macd_fast_signal', 'macd_hist': 'macd_fast_hist'
    }, errors='ignore', inplace=True)
    df = add_bollinger_bands(df, window=10)
    df = add_roc(df, [3, 5, 10])
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)
    df = add_ofi(df, window=10)
    df = add_vwap(df)
    df = add_keltner(df, ema_period=10, atr_mult=1.5, atr_period=5)

    df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)

    df['low_15'] = df['low'].rolling(15).min()
    df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']

    # Strategy signals as features
    df['signal_rsi'] = 0.0
    df.loc[df['rsi_7'] < 25, 'signal_rsi'] = 1.0
    df.loc[df['rsi_7'] > 75, 'signal_rsi'] = -1.0
    df['signal_bb'] = 0.0
    df.loc[df['bb_pb'] < 0.1, 'signal_bb'] = 1.0
    df.loc[df['bb_pb'] > 0.9, 'signal_bb'] = -1.0

    # Triple barrier: scalping uses tight barriers, 5-bar timeout
    # profit=0.5% (tight for 1m), loss=0.3%, 5 bars timeout
    labels = triple_barrier_labels_vectorized(df, profit_pct=0.005, loss_pct=0.003, max_bars=5)
    df['target_scalp'] = (labels == 1).astype(int)
    log.info("Scalping TB distribution: %s", label_stats(labels))
    return df.dropna()


def prepare_scalping_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return _engineer_scalping_features(df)


def prepare_scalping_data(filepath):
    log.info("Loading data for Scalping Pipeline from %s...", filepath)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return _engineer_scalping_features(df)


def _process_single_symbol(sym):
    full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1m.csv.gz')
    archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1m.csv.gz')
    path_1s = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1s.csv.gz')
    path_1s_v2 = os.path.join(base_dir, 'data', 'raw', f'{sym}_1s.csv.gz')

    if not os.path.exists(full_data_path) and os.path.exists(archive_path):
        full_data_path = archive_path

    df_1s_resampled = None
    for p1s in [path_1s, path_1s_v2]:
        if os.path.exists(p1s):
            try:
                log.info("  [%s] Found 1s data — resampling to 1m...", sym)
                df_1s_resampled = resample_1s_to_1m(p1s)
                if df_1s_resampled is not None and len(df_1s_resampled) > 0:
                    break
                df_1s_resampled = None
            except Exception as e:
                log.warning("  [%s] 1s resample failed: %s", sym, e)

    if not os.path.exists(full_data_path):
        if df_1s_resampled is None:
            log.warning("[%s] Data not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe='1m', days=70)

    df_combined = None
    if os.path.exists(full_data_path):
        try:
            df_1m = prepare_scalping_data(full_data_path)
        except Exception as e:
            log.warning("  [%s] Failed 1m prepare: %s", sym, e)
            df_1m = None

        if df_1m is not None and df_1s_resampled is not None:
            try:
                df_1s_feat = prepare_scalping_data_from_df(df_1s_resampled)
                df_combined = df_1s_feat if len(df_1s_feat) >= len(df_1m) else df_1m
            except Exception:
                df_combined = df_1m
        elif df_1m is not None:
            df_combined = df_1m

    if df_combined is None and df_1s_resampled is not None:
        try:
            df_combined = prepare_scalping_data_from_df(df_1s_resampled)
        except Exception as e:
            log.warning("  [%s] 1s-only feature engineering failed: %s", sym, e)

    if df_combined is not None:
        df_tail = df_combined.tail(500000).copy()
        # Downcast float64 to float32 to cut memory usage in half
        float_cols = df_tail.select_dtypes(include=['float64']).columns
        df_tail[float_cols] = df_tail[float_cols].astype('float32')
        return df_tail

    return None


def train_scalping_model():
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']

    from joblib import Parallel, delayed
    log.info("Processing 1s granular data concurrently across CPU cores...")
    # Use 2 CPU workers to double speed without triggering OOM on laptops
    results = Parallel(n_jobs=-1)(delayed(_process_single_symbol)(sym) for sym in symbols)
    
    all_data = [res for res in results if res is not None]

    if not all_data:
        log.error("No 1m data found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    for col in [f for f in FEATURE_COLUMNS if f not in combined_df.columns]:
        combined_df[col] = 0.0

    X = combined_df[FEATURE_COLUMNS].fillna(0)
    y = combined_df['target_scalp']

    log.info("Scalping dataset: %d total | features %d | symbols %s | timeframe 1m",
             len(combined_df), len(FEATURE_COLUMNS), symbols)

    splits = _walk_forward_splits(len(X), n_splits=5)
    fold_accs = []
    for i, (tr, te) in enumerate(splits):
        clf = HistGradientBoostingClassifier(
            random_state=42, max_iter=400, max_depth=5,
            learning_rate=0.05, early_stopping=True, class_weight='balanced'
        )
        clf.fit(X.iloc[tr], y.iloc[tr])
        fold_accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
        log.info("Scalping walk-forward fold %d: %.2f%%", i + 1, fold_accs[-1] * 100)

    log.info("Scalping walk-forward mean: %.2f%% ± %.2f%%",
             np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = HistGradientBoostingClassifier(
        random_state=42, max_iter=400, max_depth=5,
        learning_rate=0.05, early_stopping=True, class_weight='balanced'
    )
    base_clf.fit(X.iloc[:calib_split], y.iloc[:calib_split])
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])

    X_test = X.iloc[int(n * 0.90):]
    y_test = y.iloc[int(n * 0.90):]
    predictions = calibrated.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 400)

    log.info("Scalping Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(calibrated, os.path.join(models_dir, 'scalping_model.joblib'))

    from datetime import datetime, timezone
    with open(os.path.join(models_dir, 'scalping_model_meta.json'), 'w') as f:
        json.dump({
            "model": "Scalping (HistGBT + Calibrated)",
            "accuracy": accuracy * 100,
            "long_accuracy": long_acc, "short_accuracy": short_acc,
            "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS), "n_iterations": n_iter,
            "walk_forward_mean_acc": round(float(np.mean(fold_accs)) * 100, 2),
            "target": "triple_barrier_long_win_1m",
            "symbols": symbols, "timeframe": "1m",
            "last_trained": datetime.now(timezone.utc).isoformat()
        }, f)


if __name__ == "__main__":
    train_scalping_model()
