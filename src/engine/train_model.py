import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

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
    'signal_rsi', 'signal_macd', 'signal_bb',
]


def _walk_forward_splits(n: int, n_splits: int = 5, test_pct: float = 0.10,
                         gap_pct: float = 0.05):
    """
    Generate (train_idx, test_idx) tuples for walk-forward cross-validation.
    Each fold trains on an expanding window and tests on the next slice.
    """
    gap = int(n * gap_pct)
    test_size = int(n * test_pct)
    train_end_start = int(n * 0.50)  # first fold uses at least 50% for training

    step = (n - train_end_start - test_size - gap) // max(n_splits - 1, 1)
    splits = []
    for i in range(n_splits):
        train_end = train_end_start + i * step
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n:
            break
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


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

    # Triple Barrier labels replace binary target
    labels = triple_barrier_labels_vectorized(df, profit_pct=0.02, loss_pct=0.01, max_bars=24)
    df['target'] = labels
    # Remap to binary: treat +1 as 1 (long win), {-1, 0} as 0 (no trade / loss)
    # This preserves compatibility with existing binary classifier while encoding risk info
    df['target_tb'] = (df['target'] == 1).astype(int)

    df = df.dropna()
    stats = label_stats(df['target'])
    log.info("Triple Barrier label distribution: %s", stats)
    return df


def train_model():
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT', 'ETH_USDT']

    timeframe = '1h'
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
    splits = _walk_forward_splits(len(X), n_splits=5)
    fold_accuracies = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        base_clf = HistGradientBoostingClassifier(
            random_state=42, max_iter=500, max_depth=6,
            learning_rate=0.03, l2_regularization=0.5,
            early_stopping=True, class_weight='balanced'
        )
        base_clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_acc = accuracy_score(y.iloc[test_idx], base_clf.predict(X.iloc[test_idx]))
        fold_accuracies.append(fold_acc)
        log.info("Walk-forward fold %d/%d: accuracy=%.2f%%", fold_i + 1, len(splits), fold_acc * 100)

    log.info("Walk-forward mean accuracy: %.2f%% ± %.2f%%",
             np.mean(fold_accuracies) * 100, np.std(fold_accuracies) * 100)

    # Final model on full data with probability calibration
    log.info("Training final model with isotonic probability calibration...")
    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = HistGradientBoostingClassifier(
        random_state=42, max_iter=500, max_depth=6,
        learning_rate=0.03, l2_regularization=0.5,
        early_stopping=True, class_weight='balanced'
    )
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    base_clf.fit(X.iloc[:calib_split], y.iloc[:calib_split])
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])

    X_test = X.iloc[int(n * 0.90):]
    y_test = y.iloc[int(n * 0.90):]
    predictions = calibrated.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 500)

    log.info("Base Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'btc_rf_model.joblib')
    joblib.dump(calibrated, model_path)
    log.info("Model saved -> %s", model_path)

    from src.utils.safe_json import write_json
    from datetime import datetime, timezone
    meta_path = os.path.join(models_dir, 'btc_rf_model_meta.json')
    write_json(meta_path, {
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
    })


if __name__ == "__main__":
    train_model()
