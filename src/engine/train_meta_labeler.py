"""
Meta-Labeler Training Pipeline.

The meta-labeler is the "second pilot" — it learns to filter bad signals.
It does not predict price direction. It answers one question only:
  "Given the current market context, should I TRUST this strategy signal?"

Architecture:
  Layer 1 (Primary): any rule-based strategy generates a signal (RSI, MACD, BB, etc.)
  Layer 2 (Meta):    this model evaluates market context and outputs P(win) ∈ [0, 1]
                     Trade is blocked if P(win) < confidence_threshold (default 0.60)

Training data:
  - Run all strategies over historical OHLCV
  - Record [features_at_signal_bar, actual_trade_outcome]
  - actual_outcome = 1 if trade would have been profitable (Triple Barrier = +1), else 0

The model learns: "RSI oversold signal is reliable when OFI is positive,
                   funding is low, and regime is trending — but not when funding
                   is at extreme negative and open interest is dropping."
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_meta_labeler')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_engineering import (
    add_rsi, add_macd, add_bollinger_bands, add_roc, add_atr,
    add_ofi, add_vwap, add_funding_zscore, add_liquidity_proximity,
    add_donchian, add_keltner, add_time_features, add_taker_and_trade_features
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized

# Features the meta-labeler sees at signal time (market context)
META_FEATURES = [
    # Market regime context
    'frac_diff_d40',
    'volatility_7',       # rolling 7-bar realized vol
    'rsi_14',
    'macd_hist',
    'bb_pb',
    'ofi_z',
    'vwap_dist',
    'funding_z',
    'funding_positive',
    'liq_proximity',
    'kc_width',           # Keltner width = volatility regime
    'don_pos_20',
    'hour', 'day_of_week',
    'taker_buy_ratio',
    'atr_pct',
    # The primary strategy signal itself
    'primary_signal',     # +1 long, -1 short, 0 flat
    'signal_rsi',
    'signal_macd',
    'signal_bb',
]


def _build_signal_dataset(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Run all strategies over df, collect rows where a signal fires,
    and label each signal as won (1) or lost (0) using Triple Barrier.
    """
    df = df.copy().reset_index(drop=True)

    # Features
    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()
    df['volatility_7'] = df['return'].rolling(7).std()
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bollinger_bands(df, window=20)
    df = add_roc(df, [7])
    df = add_atr(df, 14)
    df = add_ofi(df)
    df = add_vwap(df)
    df = add_funding_zscore(df)
    df = add_liquidity_proximity(df)
    df = add_donchian(df, n=20)
    df = add_keltner(df)
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)
    df['atr_pct'] = (df['high'] - df['low']) / df['close']

    # Strategy signals
    df['signal_rsi'] = 0.0
    df.loc[df['rsi_14'] < 30, 'signal_rsi'] = 1.0
    df.loc[df['rsi_14'] > 70, 'signal_rsi'] = -1.0

    df['signal_macd'] = np.where(df['macd_hist'] > 0, 1.0, -1.0)

    df['signal_bb'] = 0.0
    df.loc[df['bb_pb'] < 0.1, 'signal_bb'] = 1.0
    df.loc[df['bb_pb'] > 0.9, 'signal_bb'] = -1.0

    # Ensemble signal
    df['primary_signal'] = (df['signal_rsi'] + df['signal_macd'] + df['signal_bb']) / 3.0

    # Triple Barrier outcome labels (2% profit, 1% loss, 24h timeout)
    tb_labels = triple_barrier_labels_vectorized(df, profit_pct=0.02, loss_pct=0.01, max_bars=24)
    df['tb_label'] = tb_labels

    # Meta-label: did the trade WIN (+1 hit upper barrier) = 1, else 0
    df['meta_label'] = (df['tb_label'] == 1).astype(int)

    # Only keep rows where at least one strategy has a non-zero signal
    signal_fired = (df['signal_rsi'].abs() > 0.1) | \
                   (df['signal_macd'].abs() > 0.1) | \
                   (df['signal_bb'].abs() > 0.1)
    df = df[signal_fired].copy()
    df['symbol'] = symbol
    return df


def train_meta_labeler():
    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    wl_path = os.path.join(PROJECT_ROOT, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

    all_frames = []
    for sym in symbols:
        loaded = False
        for fname in [f'{sym}_1h.csv.gz', f'{sym}_spot_1h.csv.gz']:
            fpath = os.path.join(raw_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                signals_df = _build_signal_dataset(df, sym)
                signals_df = signals_df.dropna(subset=META_FEATURES + ['meta_label'])
                if len(signals_df) > 50:
                    all_frames.append(signals_df)
                    log.info("%s: %d signal bars collected (win rate=%.1f%%)",
                             sym, len(signals_df),
                             signals_df['meta_label'].mean() * 100)
                    loaded = True
                    break
            except Exception as e:
                log.error("Failed to process %s: %s", sym, e)
        if not loaded:
            log.warning("No data for %s", sym)

    if not all_frames:
        log.error("No signal data collected. Cannot train meta-labeler.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    missing = [f for f in META_FEATURES if f not in combined.columns]
    for col in missing:
        combined[col] = 0.0
    if missing:
        log.warning("Missing meta features filled with 0: %s", missing)

    X = combined[META_FEATURES].fillna(0)
    y = combined['meta_label']

    log.info("Meta-labeler dataset: %d signal bars | win rate=%.1f%% | features=%d",
             len(combined), y.mean() * 100, len(META_FEATURES))

    n = len(X)
    calib_split = int(n * 0.75)
    test_start = int(n * 0.90)

    base_clf = HistGradientBoostingClassifier(
        random_state=42, max_iter=300, max_depth=4,
        learning_rate=0.05, l2_regularization=0.3,
        early_stopping=True, class_weight='balanced'
    )
    log.info("Training meta-labeler base model on %d samples...", calib_split)
    base_clf.fit(X.iloc[:calib_split], y.iloc[:calib_split])

    log.info("Calibrating probabilities with isotonic regression...")
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_split:test_start], y.iloc[calib_split:test_start])

    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    predictions = calibrated.predict(X_test)
    proba = calibrated.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.5
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    precision_win = report.get('1', {}).get('precision', 0.0) * 100

    # High-confidence trades only (threshold 0.60)
    high_conf = proba >= 0.60
    if high_conf.sum() > 0:
        hc_acc = accuracy_score(y_test[high_conf], (proba[high_conf] >= 0.60).astype(int))
        log.info("Meta-labeler | Accuracy: %.2f%% | AUC: %.3f | Win precision: %.1f%% | "
                 "High-conf trades: %d (acc=%.1f%%)",
                 accuracy * 100, auc, precision_win, high_conf.sum(), hc_acc * 100)
    else:
        log.info("Meta-labeler | Accuracy: %.2f%% | AUC: %.3f | Win precision: %.1f%%",
                 accuracy * 100, auc, precision_win)

    model_path = os.path.join(models_dir, 'meta_labeler.joblib')
    joblib.dump(calibrated, model_path)
    log.info("Meta-labeler saved -> %s", model_path)

    from datetime import datetime, timezone
    meta_path = os.path.join(models_dir, 'meta_labeler_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            "model": "Meta-Labeler (HistGBT + Calibrated)",
            "accuracy": accuracy * 100,
            "auc_roc": auc,
            "win_precision": precision_win,
            "confidence_threshold": 0.60,
            "n_samples": len(combined),
            "n_train": calib_split,
            "n_test": len(X_test),
            "n_features": len(META_FEATURES),
            "win_rate_pct": round(float(y.mean()) * 100, 1),
            "symbols": symbols,
            "timeframe": "1h",
            "last_trained": datetime.now(timezone.utc).isoformat()
        }, f)


if __name__ == "__main__":
    train_meta_labeler()
