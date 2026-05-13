"""
Meta-Labeler Training Pipeline (AFML Ch. 3 — Lopez de Prado).

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
  - actual_outcome = 1 if trade hit TP (Triple Barrier = +1), else 0
  - Timeout rows (tb_label == 0) are KEPT as negative class (NOT dropped)

AFML conformance:
  - Asymmetric Triple Barrier: pt=2.5, sl=1.5, max_bars=12 (R/R=1.67)
  - PurgedKFold CV with real t1-span purging + embargo
  - Strict temporal split: 60% train / 20% calibrate / 20% test, with purge gaps
  - CalibratedClassifierCV(method='isotonic')
  - Walk-forward Sortino threshold search [0.40, 0.70]
  - `optimal_threshold` persisted in meta JSON for runtime lookup
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
import joblib
from src.utils.purged_kfold import PurgedKFold
from src.utils.meta_config import (
    META_FEATURES,
    CONFIDENCE_THRESHOLD,
    THRESHOLD_SEARCH_RANGE,
    THRESHOLD_SEARCH_STEP,
)

# NOTE: logging.basicConfig was previously here at module-level — it silently
# reconfigured the root logger for every importer (BUG-N8). Removed.
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


def _compute_sortino_for_threshold(
    proba: np.ndarray,
    signals: np.ndarray,
    returns: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute Sortino ratio for trades approved at a given confidence threshold.

    Trades are approved when proba >= threshold AND signal != 0.
    Returns annualized Sortino (downside-only volatility penalty).
    """
    mask = (proba >= threshold) & (signals != 0)
    if mask.sum() < 10:
        return 0.0
    trade_returns = returns[mask] * signals[mask]
    mean_r = trade_returns.mean()
    downside = trade_returns[trade_returns < 0]
    if len(downside) == 0:
        return mean_r * np.sqrt(252) * 10.0  # bonus for no negatives
    downside_vol = downside.std()
    if downside_vol == 0:
        return 0.0
    return float(mean_r / downside_vol * np.sqrt(252))


def _build_signal_dataset(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Run primary models over df to generate probabilities, then label each
    potential trade as won (1) or lost (0) for meta-model training.
    """
    df = df.copy().reset_index(drop=True)
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    # --- 1. Load primary models ---
    try:
        import io
        from src.utils.model_integrity import verify_and_load_bytes
        base_path = os.path.join(models_dir, 'btc_rf_model.joblib')
        trend_path = os.path.join(models_dir, 'trend_model.joblib')
        regime_path = os.path.join(models_dir, 'regime_classifier.joblib')
        base_model = joblib.load(io.BytesIO(verify_and_load_bytes(base_path)))
        trend_model = joblib.load(io.BytesIO(verify_and_load_bytes(trend_path)))
        regime_model_data = joblib.load(io.BytesIO(verify_and_load_bytes(regime_path)))
    except FileNotFoundError as e:
        log.error("Primary models for meta-labeler not found: %s. Train them first.", e)
        return pd.DataFrame()

    # --- 2. Engineer a superset of all features ---
    from src.analysis.ml_predictor import MLPredictor
    feature_builder = MLPredictor()
    df_feat = feature_builder._build_all_features(df).copy()  # avoid SettingWithCopyWarning
    df_feat['volatility_7'] = df_feat['return'].rolling(7).std()

    # Compute signal_don from don_pos_20 if missing (trend model expects it)
    if 'don_pos_20' in df_feat.columns and 'signal_don' not in df_feat.columns:
        df_feat['signal_don'] = np.select(
            [df_feat['don_pos_20'] > 0.95, df_feat['don_pos_20'] < 0.05],
            [1.0, -1.0], default=0.0,
        )

    # Compute per-rule signal flags (rsi, macd, bb) for META_FEATURES
    if 'signal_rsi' not in df_feat.columns and 'rsi_14' in df_feat.columns:
        df_feat['signal_rsi'] = np.select(
            [df_feat['rsi_14'] < 30, df_feat['rsi_14'] > 70],
            [1.0, -1.0], default=0.0,
        )
    if 'signal_macd' not in df_feat.columns and 'macd_hist' in df_feat.columns:
        df_feat['signal_macd'] = np.sign(df_feat['macd_hist']).fillna(0.0)
    if 'signal_bb' not in df_feat.columns and 'bb_pb' in df_feat.columns:
        df_feat['signal_bb'] = np.select(
            [df_feat['bb_pb'] < 0.0, df_feat['bb_pb'] > 1.0],
            [1.0, -1.0], default=0.0,
        )
    if 'funding_positive' not in df_feat.columns and 'funding_z' in df_feat.columns:
        df_feat['funding_positive'] = (df_feat['funding_z'] > 0).astype(int)

    # --- 3. Generate primary model probabilities ---
    # Cache feature lists at module-scope to avoid 2× MLPredictor construction per symbol
    base_features = MLPredictor(model_filename='btc_rf_model.joblib')._get_model_features()
    X_base = df_feat.reindex(columns=base_features, fill_value=0).fillna(0)
    df_feat['prob_base'] = base_model.predict_proba(X_base)[:, 1]

    trend_features = MLPredictor(model_filename='trend_model.joblib')._get_model_features()
    X_trend = df_feat.reindex(columns=trend_features, fill_value=0).fillna(0)
    df_feat['prob_trend'] = trend_model.predict_proba(X_trend)[:, 1]

    # --- 4. Generate regime context ---
    from src.analysis.regime_classifier import RegimeClassifier
    regime_features_df = _compute_regime_features(df_feat)
    _rmodel = regime_model_data.get("model", regime_model_data)   # nested or flat
    X_regime_scaled = _rmodel["scaler"].transform(regime_features_df[RegimeClassifier.FEATURE_COLS])
    raw_labels = _rmodel["gmm"].predict(X_regime_scaled)
    df_feat['regime'] = pd.Series(
        [regime_model_data['label_map'].get(int(r), 0) for r in raw_labels],
        index=regime_features_df.index,
    ).ffill()

    # --- 5. AFML Triple Barrier (fixed: pt=2.5, sl=1.5, max_bars=12 — BUG-4 fix) ---
    tb_labels, t1_times = triple_barrier_labels_vectorized(
        df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12,
    )
    df_feat['tb_label'] = tb_labels
    df_feat['t1_timestamp'] = t1_times

    # BUG-4 FIX: do NOT drop timeout rows. AFML Ch.3: timeouts are negative class.
    # Previous code `df_feat = df_feat[df_feat['tb_label'] != 0]` dropped ~30%
    # of training data and broke the meta-labeler.

    # --- 6. Define primary signal and meta-label ---
    primary_signal = np.where(
        df_feat['prob_base'] > 0.52, 1,
        np.where(df_feat['prob_base'] < 0.48, -1, 0),
    )
    df_feat['primary_signal'] = primary_signal  # KEEP as column for inference parity

    # Filter to rows with an actual signal, then reset index so the comparison
    # against tb_label is positional-safe (BUG-N5: previous misalignment).
    df_feat = df_feat[primary_signal != 0].reset_index(drop=True).copy()
    primary_signal_filtered = primary_signal[primary_signal != 0]

    # meta_label = 1 if the signal direction matches the resolved barrier (TP).
    # tb_label == 0 (timeout) AND any signal direction → meta_label = 0 (correct
    # outcome to learn: "this signal expired without conviction → loss").
    df_feat['meta_label'] = (
        (np.sign(primary_signal_filtered) == df_feat['tb_label'].values)
        & (df_feat['tb_label'].values != 0)
    ).astype(int)

    df_feat['symbol'] = symbol
    return df_feat


def train_meta_labeler(timeframe: str = '1h'):
    """Train the meta-labeler classifier at a given timeframe."""
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
                log.error("Failed to process %s: %s", sym, e, exc_info=True)
        if not loaded:
            log.warning("No data for %s", sym)

    if not all_frames:
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
    # BUG-N3 fix: strict 60/20/20 temporal split with explicit purge gaps of
    # `max_bars=12` (matches the Triple Barrier window).
    purge_bars = 12
    train_end = int(n * 0.60)
    calib_start = train_end + purge_bars
    calib_end = int(n * 0.80)
    test_start = calib_end + purge_bars

    # ── Walk-forward CV on the FULL training portion only (no test leakage) ──
    _wf_fold_accs = []
    t1_series = combined['t1_timestamp']
    pct_embargo = (2.0 * 12) / len(X)  # 2× max_bars
    cv = PurgedKFold(n_splits=5, t1=t1_series.iloc[:train_end], pct_embargo=pct_embargo)
    for _fi, (tr_idx, te_idx) in enumerate(cv.split(X.iloc[:train_end])):
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

    # ── Train base classifier on train portion [0, train_end) ──
    base_clf = make_classifier(
        random_state=42, n_estimators=300, max_depth=4,
        learning_rate=0.05, l2_regularization=0.3,
        early_stopping=True, class_weight='balanced'
    )
    log.info("Training meta-labeler base on %d samples [0:%d]...", train_end, train_end)
    weights_tr = compute_sample_weight('balanced', y.iloc[:train_end])
    base_clf.fit(X.iloc[:train_end], y.iloc[:train_end], sample_weight=weights_tr)

    # ── Calibrate on [calib_start, calib_end) — never seen by base_clf ──
    log.info("Calibrating on rows [%d:%d] (isotonic, prefit)...", calib_start, calib_end)
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_start:calib_end], y.iloc[calib_start:calib_end])

    # ── Threshold search via Sortino on calibration window ──
    log.info("Searching optimal confidence threshold via Sortino ratio...")
    proba_calib = calibrated.predict_proba(X.iloc[calib_start:calib_end])[:, 1]
    primary_signals_calib = combined['primary_signal'].iloc[calib_start:calib_end].values
    returns_calib = combined['return'].iloc[calib_start:calib_end].fillna(0).values if 'return' in combined.columns else np.zeros(calib_end - calib_start)

    best_threshold = CONFIDENCE_THRESHOLD
    best_sortino = -np.inf
    threshold_grid = np.arange(THRESHOLD_SEARCH_RANGE[0], THRESHOLD_SEARCH_RANGE[1] + 1e-9, THRESHOLD_SEARCH_STEP)
    for thr in threshold_grid:
        sortino = _compute_sortino_for_threshold(
            proba_calib, primary_signals_calib, returns_calib, float(thr),
        )
        if sortino > best_sortino:
            best_sortino = sortino
            best_threshold = float(thr)
    log.info("Optimal threshold: %.3f (Sortino=%.3f)", best_threshold, best_sortino)

    # ── Final test on [test_start, end) ──
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    predictions = calibrated.predict(X_test)
    proba = calibrated.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.5
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    precision_win = report.get('1', {}).get('precision', 0.0) * 100

    # High-confidence trades only — use the OPTIMAL threshold found above
    high_conf = proba >= best_threshold
    if high_conf.sum() > 0:
        hc_acc = accuracy_score(y_test[high_conf], (proba[high_conf] >= best_threshold).astype(int))
        log.info("Meta-labeler | Accuracy: %.2f%% | AUC: %.3f | Win precision: %.1f%% | "
                 "High-conf trades (thr=%.2f): %d (acc=%.1f%%)",
                 accuracy * 100, auc, precision_win, best_threshold, high_conf.sum(), hc_acc * 100)
    else:
        log.info("Meta-labeler | Accuracy: %.2f%% | AUC: %.3f | Win precision: %.1f%%",
                 accuracy * 100, auc, precision_win)

    # ── Persist via canonical model_paths helper ──
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    from src.utils.model_integrity import sign_model
    paths = artifact_paths('meta', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    sign_model(str(paths['model']))
    log.info("Meta-labeler saved -> %s", paths['model'])

    meta = {
        "model": "Meta-Labeler (HistGBT + Calibrated, AFML asymmetric barriers)",
        "accuracy": accuracy * 100,
        "auc_roc": auc,
        "win_precision": precision_win,
        "confidence_threshold": best_threshold,    # legacy alias for older code
        "optimal_threshold":    best_threshold,    # new canonical key
        "optimal_sortino":      round(best_sortino, 4),
        "n_samples": len(combined),
        "n_train":   train_end,
        "n_calib":   calib_end - calib_start,
        "n_test":    len(X_test),
        "n_features": len(META_FEATURES),
        "meta_features": list(META_FEATURES),       # canonical feature list snapshot
        "win_rate_pct": round(float(y.mean()) * 100, 1),
        "symbols": symbols,
        "timeframe": timeframe,
        "walk_forward_mean_acc": round(_wf_mean_acc, 2),
        "walk_forward_std_acc":  round(_wf_std_acc,  2),
        "walk_forward_folds":    len(_wf_fold_accs),
        "afml_params": {"pt": 2.5, "sl": 1.5, "max_bars": 12},
        "last_trained": datetime.now(timezone.utc).isoformat(),
    }
    write_json(str(paths['meta']), meta)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        sign_model(str(paths['legacy_model']))
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)


if __name__ == "__main__":
    # Configure logging only when running as a script (not at import time).
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    ap = argparse.ArgumentParser(description="Train the meta-labeler")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_meta_labeler(timeframe=args.timeframe)
