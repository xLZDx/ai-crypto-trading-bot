import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from mlfinlab.cross_validation import PurgedKFold
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_trend')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_macd, add_adx, add_time_features, add_atr,
    add_ofi, add_vwap, add_donchian, add_keltner, add_funding_zscore
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats


FEATURE_COLUMNS = [
    'frac_diff_d40',
    'macd', 'macd_signal', 'macd_hist',
    'trend_alignment',
    'volume_surge',
    'atr_14', 'adx_14',
    'don_pos_20',
    'kc_pos',
    'kc_width',
    'vwap_dist',
    'ofi_z',
    'funding_z',
    'signal_macd', 'signal_don',
    'trend_strength',
    'vol_regime',
    'is_trending',
    'is_volatile',
]


def prepare_trend_data(filepath):
    log.info("Loading data for Trend Pipeline from %s...", filepath)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()
    df = add_macd(df)
    df = add_adx(df, period=14)
    df = add_time_features(df)
    df = add_ofi(df)
    df = add_vwap(df)
    df = add_donchian(df, n=20)
    df = add_keltner(df)
    df = add_funding_zscore(df)
    df = add_atr(df, 14) # Ensure atr_14 exists

    # Regime Features
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['trend_strength'] = (df['ema_fast'] - df['ema_slow']).abs() / df['atr_14'].replace(0, 1e-9)
    df['vol_short'] = df['return'].rolling(window=7).std()
    df['vol_long'] = df['return'].rolling(window=100, min_periods=10).std()
    df['vol_regime'] = df['vol_short'] / df['vol_long'].replace(0, 1e-9)
    df['is_trending'] = (df['trend_strength'] > 1.5).astype(int)
    df['is_volatile'] = (df['vol_regime'] > 1.5).astype(int)

    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['trend_alignment'] = (df['sma_50'] > df['sma_200']).astype(int)

    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_20'] * 1.5).astype(int)

    df['signal_macd'] = np.where(df['macd_hist'] > 0, 1.0, -1.0)
    df['signal_don'] = 0.0
    df.loc[df['don_pos_20'] > 0.95, 'signal_don'] = 1.0
    df.loc[df['don_pos_20'] < 0.05, 'signal_don'] = -1.0

    labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=3.0, sl_multiplier=3.0, max_bars=48)
    df['target_raw'] = labels
    df['t1_timestamp'] = t1_times
    
    # Remove timeouts
    df = df[df['target_raw'] != 0].copy()
    df['target'] = (df['target_raw'] == 1).astype(int)
    
    df = df.dropna()
    log.info("Trend TB distribution: %s", label_stats(labels))
    return df


def train_trend_model():
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']

    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1h.csv.gz')
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1h.csv.gz')
        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
        if not os.path.exists(full_data_path):
            log.warning("Data for %s not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe='1h', days=6 * 365)

        if os.path.exists(full_data_path):
            try:
                df = prepare_trend_data(full_data_path)
                all_data.append(df)
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
    y = combined_df['target']

    log.info("Trend dataset: %d total | features %d | symbols %s | timeframe 1h",
             len(combined_df), len(FEATURE_COLUMNS), symbols)

    t1_series = combined_df['t1_timestamp']
    # Embargo = 2 * horizon (48 bars for trend model)
    pct_embargo = (2.0 * 48) / len(X)
    cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=pct_embargo)
    fold_accs = []
    for i, (tr, te) in enumerate(cv.split(X)):
        clf = HistGradientBoostingClassifier(
            random_state=42, max_iter=500, max_depth=5,
            learning_rate=0.02, early_stopping=True, l2_regularization=0.5, class_weight='balanced'
        )
        weights = compute_sample_weight('balanced', y.iloc[tr])
        clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights)
        fold_accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
        log.info("Trend walk-forward fold %d: %.2f%%", i + 1, fold_accs[-1] * 100)

    log.info("Trend walk-forward mean: %.2f%% ± %.2f%%",
             np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = HistGradientBoostingClassifier(
        random_state=42, max_iter=500, max_depth=5,
        learning_rate=0.02, early_stopping=True, l2_regularization=0.5, class_weight='balanced'
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
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 500)

    log.info("Trend Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(calibrated, os.path.join(models_dir, 'trend_model.joblib'))

    from datetime import datetime, timezone
    with open(os.path.join(models_dir, 'trend_model_meta.json'), 'w') as f:
        json.dump({
            "model": "Trend (HistGBT + Calibrated)",
            "accuracy": accuracy * 100,
            "long_accuracy": long_acc, "short_accuracy": short_acc,
            "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS), "n_iterations": n_iter,
            "walk_forward_mean_acc": round(float(np.mean(fold_accs)) * 100, 2),
            "target": "triple_barrier_long_win",
            "symbols": symbols, "timeframe": "1h",
            "last_trained": datetime.now(timezone.utc).isoformat()
        }, f)


if __name__ == "__main__":
    train_trend_model()
