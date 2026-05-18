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
from src.data_ingestion.ohlcv_parquet_loader import load_ohlcv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
import joblib
from src.utils.purged_kfold import PurgedKFold
from src.engine.kpi_gate import hard_gate_wf, KPIGateFailure
from src.utils.sample_weights import compute_afml_weights
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
    add_explicit_regime,
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


def _build_signal_dataset(
    df: pd.DataFrame,
    symbol: str,
    pt_multiplier: float = 2.5,
    sl_multiplier: float = 1.5,
    max_bars: int = 12,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Run primary models over df to generate probabilities, then label each
    potential trade as won (1) or lost (0) for meta-model training.

    Triple Barrier params can be overridden by the caller (typically via
    CIO operator-approved overrides flowing through train_meta_labeler()).
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
    df_feat = feature_builder._build_all_features(df, symbol=symbol, timeframe=timeframe).copy()
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
        df, pt_multiplier=pt_multiplier, sl_multiplier=sl_multiplier, max_bars=max_bars,
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


def _load_cio_overrides(model_key: str = 'meta') -> dict:
    """Thin wrapper around the shared helper for backwards compatibility."""
    from src.utils.cio_overrides import load_cio_overrides as _shared
    return _shared(model_key)


def train_meta_labeler(
    timeframe: str = '1h',
    pt_multiplier: float | None = None,
    sl_multiplier: float | None = None,
    max_bars: int | None = None,
    confidence_threshold: float | None = None,
):
    """Train the meta-labeler classifier at a given timeframe.

    Any HP arg left as None is filled from `data/training_rules.json`
    `models.meta.cio_overrides` (if present) — this is how operator-approved
    CIO Agent proposals reach the trainer. If neither caller nor cio_overrides
    set a value, the AFML defaults baked into the function bodies are used.
    """
    # ── CIO overrides resolution ──────────────────────────────────────────
    cio = _load_cio_overrides('meta')
    pt = pt_multiplier if pt_multiplier is not None else cio.get('pt_multiplier', 2.5)
    sl = sl_multiplier if sl_multiplier is not None else cio.get('sl_multiplier', 1.5)
    mb = max_bars      if max_bars      is not None else cio.get('max_bars',      12)
    thr_hint = (confidence_threshold
                if confidence_threshold is not None
                else cio.get('confidence_threshold'))
    # `timeframe` is honored from the CLI/cluster spec only — overriding it from
    # cio_overrides would silently swap which (model, tf) cell we're training.
    if cio:
        log.info("[CIO overrides] applied to meta: pt=%.2f sl=%.2f max_bars=%d "
                 "confidence_hint=%s (study=%s, best_value=%s)",
                 pt, sl, mb, thr_hint, cio.get('_study') or '?', cio.get('_best_value') or '?')

    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    wl_path = os.path.join(PROJECT_ROOT, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

    _syms_sorted = sorted(symbols)
    all_frames = []
    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe)
            if df.empty:
                log.warning("No data for %s", sym)
                continue
            # Phase 4 rollout — F1 data-integrity gate.
            try:
                from src.utils.data_quality import validate_ohlcv, DataQualityError
                df, _dq = validate_ohlcv(df, symbol=sym, timeframe=timeframe)
                if _dq.soft_warnings:
                    log.info("[meta][%s/%s] data quality: %s",
                             sym, timeframe, _dq.soft_warnings[:3])
            except Exception as _e:
                from src.utils.data_quality import DataQualityError as _DQE
                if isinstance(_e, _DQE):
                    raise
                log.warning("[meta][%s/%s] data_quality check skipped: %s",
                            sym, timeframe, _e)
            signals_df = _build_signal_dataset(
                df, sym,
                pt_multiplier=pt, sl_multiplier=sl, max_bars=mb,
                timeframe=timeframe,
            )
            signals_df = add_explicit_regime(signals_df)
            signals_df['symbol_id'] = float(_syms_sorted.index(sym) + 1)
            signals_df = signals_df.dropna(subset=META_FEATURES + ['meta_label'])
            if len(signals_df) > 50:
                all_frames.append(signals_df)
                log.info("%s: %d signal bars collected (win rate=%.1f%%)",
                         sym, len(signals_df),
                         signals_df['meta_label'].mean() * 100)
            else:
                log.warning("No data for %s", sym)
        except Exception as e:
            log.error("Failed to process %s: %s", sym, e, exc_info=True)

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
    _wf_in_sample_fold_accs = []  # P3
    t1_series = combined['t1_timestamp']
    cv = PurgedKFold(n_splits=5, t1=t1_series.iloc[:train_end], embargo_td=pd.Timedelta(hours=24))
    for _fi, (tr_idx, te_idx) in enumerate(cv.split(X.iloc[:train_end])):
        _wf_clf = make_classifier(
            random_state=42, n_estimators=200, max_depth=4,
            learning_rate=0.05, l2_regularization=0.3, class_weight='balanced'
        )
        weights = compute_afml_weights(y.iloc[tr_idx], t1_series.iloc[tr_idx], combined['return'].iloc[tr_idx])
        _wf_clf.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=weights)
        _fold_acc = accuracy_score(y.iloc[te_idx], _wf_clf.predict(X.iloc[te_idx]))
        _wf_fold_accs.append(_fold_acc)
        _wf_in_sample_fold_accs.append(
            accuracy_score(y.iloc[tr_idx], _wf_clf.predict(X.iloc[tr_idx]))
        )
        log.info("Meta-labeler WF fold %d/5: accuracy=%.2f%%", _fi + 1, _fold_acc * 100)
    _wf_mean_acc = float(np.mean(_wf_fold_accs)) * 100 if _wf_fold_accs else 0.0
    _wf_std_acc  = float(np.std(_wf_fold_accs))  * 100 if _wf_fold_accs else 0.0
    log.info("Meta-labeler WF mean accuracy: %.2f%% +/- %.2f%%", _wf_mean_acc, _wf_std_acc)

    # P3: overfit ratio
    _wf_mean_frac = _wf_mean_acc / 100.0
    _in_sample_mean_meta = float(np.mean(_wf_in_sample_fold_accs)) if _wf_in_sample_fold_accs else None
    _overfit_ratio_meta: float | None = None
    if _in_sample_mean_meta is not None and _in_sample_mean_meta > 0:
        _overfit_ratio_meta = (_in_sample_mean_meta - _wf_mean_frac) / _in_sample_mean_meta
        if _overfit_ratio_meta > 0.20:
            log.error("[meta] overfit_ratio=%.3f > 0.20 (in_sample=%.2f%% wf=%.2f%%) -- model is memorising",
                      _overfit_ratio_meta, _in_sample_mean_meta * 100, _wf_mean_acc)
        elif _overfit_ratio_meta > 0.10:
            log.warning("[meta] overfit_ratio=%.3f > 0.10 (in_sample=%.2f%% wf=%.2f%%)",
                        _overfit_ratio_meta, _in_sample_mean_meta * 100, _wf_mean_acc)

    hard_gate_wf(_wf_mean_frac, 'meta')

    # ── Train base classifier on train portion [0, train_end) ──
    base_clf = make_classifier(
        random_state=42, n_estimators=300, max_depth=4,
        learning_rate=0.05, l2_regularization=0.3,
        early_stopping=True, class_weight='balanced'
    )
    log.info("Training meta-labeler base on %d samples [0:%d]...", train_end, train_end)
    weights_tr = compute_afml_weights(y.iloc[:train_end], t1_series.iloc[:train_end], combined['return'].iloc[:train_end])
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

    # High-confidence trades only — use the OPTIMAL threshold found above.
    # Compare the model's actual predictions on the high-conf subset against
    # the true labels (was a tautology before — proba[hc] >= thr is always True
    # by definition, so the old "accuracy" was just the positive-class rate).
    high_conf = proba >= best_threshold
    if high_conf.sum() > 0:
        hc_predictions = predictions[high_conf]  # actual model predictions on hc subset
        hc_acc = accuracy_score(y_test[high_conf], hc_predictions)
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

    # Phase 6b wire-in — persist training feature distribution baseline.
    try:
        from src.risk.drift_baseline import save_baseline
        save_baseline('meta', timeframe, X)
    except Exception as _e:
        log.warning("[meta][%s] save_baseline failed: %s", timeframe, _e)

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
        "symbols": symbols, "symbols_sorted": _syms_sorted,
        "timeframe": timeframe,
        "walk_forward_mean_acc": round(_wf_mean_acc, 2),
        "walk_forward_std_acc":  round(_wf_std_acc,  2),
        "walk_forward_folds":    len(_wf_fold_accs),
        "wf_fold_scores": [round(v, 6) for v in _wf_fold_accs],
        "in_sample_mean_acc": round(_in_sample_mean_meta * 100, 2) if _in_sample_mean_meta else None,
        "overfit_ratio": round(_overfit_ratio_meta, 6) if _overfit_ratio_meta is not None else None,
        "afml_params": {"pt": float(pt), "sl": float(sl), "max_bars": int(mb)},
        "cio_overrides_applied": dict(cio) if cio else None,
        "last_trained": datetime.now(timezone.utc).isoformat(),
    }
    write_json(str(paths['meta']), meta)
    try:
        from src.engine.champion_challenger import ChampionRegistry
        meta['model_path'] = str(paths['model'])
        ChampionRegistry().register_challenger('meta', timeframe, meta)
    except Exception as _cc_e:
        log.warning("[meta] champion_challenger failed: %s", _cc_e)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        sign_model(str(paths['legacy_model']))
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)

    # Phase K (2026-05-14) — record run in training history.
    try:
        from src.analytics.training_history import record_run_from_meta
        record_run_from_meta(meta, model='meta', tf=timeframe,
                             trainer='train_meta_labeler.py',
                             meta_path=str(paths['meta']))
    except Exception as e:
        log.warning("[meta] record_run skipped: %s", e)


if __name__ == "__main__":
    # Configure logging only when running as a script (not at import time).
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    ap = argparse.ArgumentParser(description="Train the meta-labeler")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_meta_labeler(timeframe=args.timeframe)
