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
from sklearn.ensemble import HistGradientBoostingClassifier  # kept for type compat
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
import joblib
from src.utils.purged_kfold import PurgedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_meta_labeler')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_engineering import (
    add_rsi, add_macd, add_bollinger_bands, add_roc, add_atr,
    add_ofi, add_vwap, add_funding_zscore, add_liquidity_proximity,
    add_donchian, add_keltner, add_time_features, add_taker_and_trade_features,
)
from src.analysis.regime_classifier import _compute_regime_features
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized

# Features the meta-labeler sees at signal time (market context)
META_FEATURES = [
    # Market regime context
    'prob_base',
    'prob_trend',
    'regime',
    'volatility_7',       # rolling 7-bar realized vol
    'ofi_z',
    'vwap_dist',
    'funding_z',
    'liq_proximity',
    'kc_width',           # Keltner width = volatility regime
    'hour', 'day_of_week',
    'taker_buy_ratio',
    'atr_pct',
]


def _build_signal_dataset(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Run primary models over df to generate probabilities, then label each
    potential trade as won (1) or lost (0) for meta-model training.
    """
    df = df.copy().reset_index(drop=True)
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    # --- 1. Load primary models ---
    try:
        base_model = joblib.load(os.path.join(models_dir, 'btc_rf_model.joblib'))
        trend_model = joblib.load(os.path.join(models_dir, 'trend_model.joblib'))
        regime_model_data = joblib.load(os.path.join(models_dir, 'regime_classifier.joblib'))
    except FileNotFoundError as e:
        log.error("Primary models for meta-labeler not found: %s. Train them first.", e)
        return pd.DataFrame()

    # --- 2. Engineer a superset of all features ---
    from src.analysis.ml_predictor import MLPredictor
    feature_builder = MLPredictor()
    df_feat = feature_builder._build_all_features(df)
    df_feat['volatility_7'] = df_feat['return'].rolling(7).std()

    # 2026-05-10 — Bug #2 fix. The trend model was trained with
    # `signal_don` as a feature (computed from `don_pos_20` inside
    # train_trend_model.py), but _build_all_features doesn't produce
    # it — so column selection at line below raised
    # KeyError: "['signal_don'] not in index" for every symbol when the
    # meta-labeler tried to apply the trend model. Compute it here from
    # don_pos_20 (which IS produced by _build_all_features) so the
    # trend model has its full feature set.
    if 'don_pos_20' in df_feat.columns and 'signal_don' not in df_feat.columns:
        df_feat['signal_don'] = 0.0
        df_feat.loc[df_feat['don_pos_20'] > 0.95, 'signal_don'] = 1.0
        df_feat.loc[df_feat['don_pos_20'] < 0.05, 'signal_don'] = -1.0

    # --- 3. Generate primary model probabilities ---
    # Defensive: use reindex(fill_value=0) instead of bare [cols] so a
    # future model trained with a feature missing from _build_all_features
    # doesn't crash the meta-labeler — degrades gracefully to a 0-valued
    # feature instead. Pair with explicit feature computation above for
    # known cases (signal_don) to preserve accuracy.
    base_features = MLPredictor(model_filename='btc_rf_model.joblib')._get_model_features()
    X_base = df_feat.reindex(columns=base_features, fill_value=0).fillna(0)
    df_feat['prob_base'] = base_model.predict_proba(X_base)[:, 1]

    trend_features = MLPredictor(model_filename='trend_model.joblib')._get_model_features()
    X_trend = df_feat.reindex(columns=trend_features, fill_value=0).fillna(0)
    df_feat['prob_trend'] = trend_model.predict_proba(X_trend)[:, 1]

    # --- 4. Generate regime context ---
    from src.analysis.regime_classifier import RegimeClassifier
    regime_features_df = _compute_regime_features(df_feat)
    X_regime_scaled = regime_model_data["scaler"].transform(regime_features_df[RegimeClassifier.FEATURE_COLS])
    raw_labels = regime_model_data["gmm"].predict(X_regime_scaled)
    df_feat['regime'] = pd.Series([regime_model_data['label_map'].get(int(r), 0) for r in raw_labels], index=regime_features_df.index).ffill()

    # Dynamic Triple Barrier outcome labels
    tb_labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=2.0, sl_multiplier=2.0, max_bars=24)
    df_feat['tb_label'] = tb_labels
    df_feat['t1_timestamp'] = t1_times
    
    # Remove timeouts to enforce strict binary classification on resolved trades
    df_feat = df_feat[df_feat['tb_label'] != 0].copy()

    # --- 6. Define primary signal and meta-label ---
    primary_signal = np.where(df_feat['prob_base'] > 0.52, 1, np.where(df_feat['prob_base'] < 0.48, -1, 0))
    df_feat = df_feat[primary_signal != 0].copy()
    primary_signal_filtered = primary_signal[primary_signal != 0]
    df_feat['meta_label'] = (np.sign(primary_signal_filtered) == df_feat['tb_label']).astype(int)

    df_feat['symbol'] = symbol
    return df_feat


def train_meta_labeler(timeframe: str = '1h'):
    """Train the meta-labeler classifier at a given timeframe. The
    meta-labeler operates on signals from primary strategies/models, so
    its TF should match the primary it filters (default 1h matches the
    base/trend/futures models)."""
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
        for fname in [f'{sym}_{timeframe}.csv.gz', f'{sym}_spot_{timeframe}.csv.gz']:
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
        # 2026-05-10 — was a silent success: function returned None, the
        # cluster orchestrator marked the task `done`, the dashboard kept
        # showing the meta model as STALE because no artifact was written.
        # Hard-fail now so the worker reports the task as failed and the
        # operator sees the real cause (primary models can't generate
        # signals on the loaded data — usually feature mismatch from
        # sklearn-version drift or a feature-engineering regression).
        msg = ("meta-labeler: no signal data collected from any symbol — "
               "primary models (base/trend) couldn't generate signals on "
               "the loaded OHLCV. Check that primaries are fresh and that "
               "feature_engineering produces all META_FEATURES.")
        log.error(msg)
        raise RuntimeError(msg)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values('timestamp')
    combined.set_index('timestamp', inplace=True)

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

    _wf_fold_accs = []
    t1_series = combined['t1_timestamp']
    # Embargo = 2 * horizon (24 bars for meta model)
    pct_embargo = (2.0 * 24) / len(X)
    cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=pct_embargo)
    for _fi, (tr_idx, te_idx) in enumerate(cv.split(X)):
        _wf_clf = make_classifier(
            random_state=42, n_estimators=200, max_depth=4,
            learning_rate=0.05, l2_regularization=0.3, class_weight='balanced'
        )
        weights = compute_sample_weight('balanced', y.iloc[tr_idx])
        _wf_clf.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=weights)
        _fold_acc = accuracy_score(y.iloc[te_idx], _wf_clf.predict(X.iloc[te_idx]))
        _wf_fold_accs.append(_fold_acc)
        log.info("Meta-labeler WF fold %d/5: accuracy=%.2f%%", _fi + 1, _fold_acc * 100)
    _wf_mean_acc = float(np.mean(_wf_fold_accs)) * 100 if _wf_fold_accs else 0.0
    _wf_std_acc  = float(np.std(_wf_fold_accs))  * 100 if _wf_fold_accs else 0.0
    log.info("Meta-labeler WF mean accuracy: %.2f%% ± %.2f%%", _wf_mean_acc, _wf_std_acc)

    base_clf = make_classifier(
        random_state=42, n_estimators=300, max_depth=4,
        learning_rate=0.05, l2_regularization=0.3,
        early_stopping=True, class_weight='balanced'
    )
    log.info("Training meta-labeler base model on %d samples...", calib_split)
    calib_start_time = combined.index[calib_split]
    valid_train_mask = combined['t1_timestamp'].iloc[:calib_split] < calib_start_time
    safe_train_idx = np.arange(calib_split)[valid_train_mask]
    weights_calib = compute_sample_weight('balanced', y.iloc[safe_train_idx])
    base_clf.fit(X.iloc[safe_train_idx], y.iloc[safe_train_idx], sample_weight=weights_calib)

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

    # ── Persist via canonical model_paths helper ──────────────────────────
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    paths = artifact_paths('meta', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    log.info("Meta-labeler saved -> %s", paths['model'])

    meta = {
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
        "timeframe": timeframe,
        "walk_forward_mean_acc": round(_wf_mean_acc, 2),
        "walk_forward_std_acc":  round(_wf_std_acc,  2),
        "walk_forward_folds":    len(_wf_fold_accs),
        "last_trained": datetime.now(timezone.utc).isoformat()
    }
    write_json(str(paths['meta']), meta)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the meta-labeler")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_meta_labeler(timeframe=args.timeframe)
