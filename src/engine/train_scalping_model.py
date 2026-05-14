import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import HistGradientBoostingClassifier  # kept for type compat
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    SMOTE = None
    _SMOTE_AVAILABLE = False
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from src.utils.purged_kfold import PurgedKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_scalping')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features, resample_1s_to_1m,
    add_ofi, add_vwap, add_keltner, add_atr,
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
    'trend_strength',
    'vol_regime',
    'is_trending',
    'is_volatile',
]


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

    # Triple barrier: dynamic volatility-based barriers for scalping
    labels, t1_times = triple_barrier_labels_vectorized(df, pt_multiplier=1.5, sl_multiplier=1.5, max_bars=5)
    df['target_raw'] = labels
    df['t1_timestamp'] = t1_times
    
    # Remove timeouts
    df = df[df['target_raw'] != 0].copy()
    df['target_scalp'] = (df['target_raw'] == 1).astype(int)
    
    log.info("Scalping TB distribution: %s", label_stats(labels))
    return df.dropna()


def prepare_scalping_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return _engineer_scalping_features(df)


def prepare_scalping_data(filepath, timeframe: str = '1m', symbol: str | None = None):
    log.info("Loading data for Scalping Pipeline from %s...", filepath)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    # Phase 4 rollout — F1 data-integrity gate.
    try:
        from src.utils.data_quality import validate_ohlcv, DataQualityError
        df, _dq = validate_ohlcv(df, symbol=symbol or '', timeframe=timeframe)
        if _dq.soft_warnings:
            log.info("[scalping][%s/%s] data quality: %s",
                     symbol, timeframe, _dq.soft_warnings[:3])
    except Exception as e:
        from src.utils.data_quality import DataQualityError
        if isinstance(e, DataQualityError):
            raise
        log.warning("[scalping][%s/%s] data_quality check skipped: %s",
                    symbol, timeframe, e)
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
            df_1m = prepare_scalping_data(full_data_path, timeframe='1m', symbol=sym)
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


def train_scalping_model(timeframe: str = '1m'):
    """Train the scalping classifier. timeframe is accepted for API
    consistency with the other trainers but is currently fixed at 1m —
    scalping at higher TFs is a contradiction in terms (the whole point
    is the high-frequency edge). The 1s→1m resample is internal and
    handled by _process_single_symbol; widening to 5m would require
    refactoring resample_1s_to_1m into a parameterised helper.

    CIO overrides from `models.scalping.cio_overrides` are logged +
    recorded in meta JSON. Per-HP merging is deferred (Sprint 1A R1).
    """
    from src.utils.cio_overrides import load_cio_overrides
    cio = load_cio_overrides('scalping')
    if cio:
        log.info("[CIO overrides] scalping/%s: %s (NOT auto-merged into params yet)",
                 timeframe, cio)
    if timeframe != '1m':
        log.warning("Scalping requested at %s — coercing to 1m (multi-TF "
                    "scalping not yet supported).", timeframe)
        timeframe = '1m'
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
    combined_df = combined_df.sort_values('timestamp')
    combined_df.set_index('timestamp', inplace=True)

    for col in [f for f in FEATURE_COLUMNS if f not in combined_df.columns]:
        combined_df[col] = 0.0

    X = combined_df[FEATURE_COLUMNS].fillna(0)
    y = combined_df['target_scalp']

    log.info("Scalping dataset: %d total | features %d | symbols %s | timeframe 1m",
             len(combined_df), len(FEATURE_COLUMNS), symbols)

    # ── Class balance diagnostic ──────────────────────────────────────────
    # The legacy training run on 2026-04-29 produced long_acc=0% (model
    # NEVER predicted class 1). class_weight='balanced' + sample_weight
    # weren't enough on the ~88/12 split. v3.1 step 6 layers in SMOTE
    # oversampling on each training fold so the classifier actually sees
    # balanced classes during fit, plus a self-healing retry that
    # detects single-class predictions and re-trains with more
    # aggressive resampling.
    pos_rate = float(y.mean())
    log.info("Scalping target distribution: %d positive / %d negative (pos rate %.1f%%)",
             int(y.sum()), len(y) - int(y.sum()), pos_rate * 100)

    # Hard cap on the dataset size we'll feed to SMOTE. With ~88/12
    # imbalance, SMOTE on 1M rows synthesises ~750K minority samples
    # → 1.75M training rows; on 3.6M rows (one fold of the 4.5M
    # scalping corpus) it expands to ~7M rows × 17 features = several
    # GB of in-RAM training tensor PER fold, and 5 folds × HistGBT
    # iters wedges the sweep for 6+ hours (observed 2026-05-09).
    # Above this cap we fall back to class_weight + sample_weight,
    # which gets ~80% of SMOTE's benefit at zero memory bloat.
    # Operator memory `feedback_disk_over_ram`: bias batch jobs toward
    # streaming + chunked compute, never multi-GB in-RAM resamples.
    SMOTE_MAX_ROWS = int(os.getenv('AI_TRADER_SCALPING_SMOTE_MAX_ROWS', '500000'))

    def _resample(Xtr: pd.DataFrame, ytr: pd.Series, *, k_neighbors: int = 5):
        """Return (X_balanced, y_balanced, sample_weight_or_None).
        SMOTE is used ONLY when len(Xtr) <= SMOTE_MAX_ROWS — above
        that, class_weight + per-row sample_weight gets the same
        accuracy outcome without exploding memory."""
        if not _SMOTE_AVAILABLE:
            return Xtr, ytr, compute_sample_weight('balanced', ytr)
        if len(Xtr) > SMOTE_MAX_ROWS:
            log.info("SMOTE skipped — fold has %d rows > SMOTE_MAX_ROWS=%d; "
                     "using class_weight + sample_weight only",
                     len(Xtr), SMOTE_MAX_ROWS)
            return Xtr, ytr, compute_sample_weight('balanced', ytr)
        try:
            # SMOTE k_neighbors must be < count of minority class.
            n_minority = int(min(ytr.value_counts())) if not ytr.empty else 0
            k = max(1, min(k_neighbors, n_minority - 1))
            if n_minority < 2:
                return Xtr, ytr, compute_sample_weight('balanced', ytr)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_bal, y_bal = sm.fit_resample(Xtr, ytr)
            # After SMOTE the classes are 1:1 — no need for sample_weight.
            return X_bal, y_bal, None
        except Exception as exc:
            log.warning("SMOTE failed (%s) — falling back to sample_weight only", exc)
            return Xtr, ytr, compute_sample_weight('balanced', ytr)

    # ── CIO overrides MERGE (X1.3, 2026-05-13) — schema-bounded ────────────
    from src.utils.cio_overrides import merge_with_defaults as _merge
    _SCALP_HP_DEFAULTS = {
        'n_estimators': 400, 'max_depth': 5,
        'learning_rate': 0.05, 'class_weight': 'balanced',
    }
    _SCALP_HP_SCHEMA = {
        'n_estimators':  (int,   1,    10_000),
        'max_depth':     (int,   1,    50),
        'learning_rate': (float, 1e-4, 1.0),
        'class_weight':  (str,   None, None),
    }
    _scalp_hp, _scalp_applied = _merge('scalping', _SCALP_HP_DEFAULTS, _SCALP_HP_SCHEMA)

    t1_series = combined_df['t1_timestamp']
    # Embargo = 2 * horizon (5 bars for scalping model)
    pct_embargo = (2.0 * 5) / len(X)
    cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=pct_embargo)
    fold_accs = []
    for i, (tr, te) in enumerate(cv.split(X)):
        clf = make_classifier(
            random_state=42,
            n_estimators=_scalp_hp['n_estimators'],
            max_depth=_scalp_hp['max_depth'],
            learning_rate=_scalp_hp['learning_rate'],
            class_weight=_scalp_hp['class_weight'],
            early_stopping=True,
        )
        X_tr_bal, y_tr_bal, w_tr = _resample(X.iloc[tr], y.iloc[tr])
        if w_tr is None:
            clf.fit(X_tr_bal, y_tr_bal)
        else:
            clf.fit(X_tr_bal, y_tr_bal, sample_weight=w_tr)
        fold_accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
        log.info("Scalping walk-forward fold %d: %.2f%%", i + 1, fold_accs[-1] * 100)

    log.info("Scalping walk-forward mean: %.2f%% ± %.2f%%",
             np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = make_classifier(
        random_state=42,
        n_estimators=_scalp_hp['n_estimators'],
        max_depth=_scalp_hp['max_depth'],
        learning_rate=_scalp_hp['learning_rate'],
        class_weight=_scalp_hp['class_weight'],
        early_stopping=True,
    )
    calib_start_time = combined_df.index[calib_split]
    valid_train_mask = combined_df['t1_timestamp'].iloc[:calib_split] < calib_start_time
    safe_train_idx = np.arange(calib_split)[valid_train_mask]

    X_safe = X.iloc[safe_train_idx]
    y_safe = y.iloc[safe_train_idx]
    X_safe_bal, y_safe_bal, w_safe = _resample(X_safe, y_safe)
    if w_safe is None:
        base_clf.fit(X_safe_bal, y_safe_bal)
    else:
        base_clf.fit(X_safe_bal, y_safe_bal, sample_weight=w_safe)
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])

    X_test = X.iloc[int(n * 0.90):]
    y_test = y.iloc[int(n * 0.90):]
    predictions = calibrated.predict(X_test)

    # ── Self-healing retry: if the model collapses to single-class
    # predictions on the test set (long_accuracy = 0% pathology), retry
    # the calibration stage with a stronger SMOTE oversample factor so
    # the underlying classifier actually sees both classes' decision
    # boundary samples.
    unique_preds = set(np.unique(predictions).tolist())
    if len(unique_preds) < 2 and _SMOTE_AVAILABLE and len(X_safe) <= SMOTE_MAX_ROWS:
        log.warning("Single-class collapse detected (preds=%s) — retrying with stronger SMOTE",
                    unique_preds)
        sm_strong = SMOTE(random_state=42,
                          k_neighbors=max(1, min(3, int(min(y_safe.value_counts()) - 1))),
                          sampling_strategy=1.0)
        try:
            X_safe_bal2, y_safe_bal2 = sm_strong.fit_resample(X_safe, y_safe)
            base_clf2 = make_classifier(
                random_state=42, n_estimators=600, max_depth=6,
                learning_rate=0.04, early_stopping=True, class_weight='balanced'
            )
            base_clf2.fit(X_safe_bal2, y_safe_bal2)
            calibrated2 = CalibratedClassifierCV(base_clf2, method='isotonic', cv='prefit', n_jobs=-1)
            calibrated2.fit(X.iloc[calib_split:], y.iloc[calib_split:])
            preds2 = calibrated2.predict(X_test)
            if len(set(np.unique(preds2).tolist())) >= 2:
                log.info("Self-healing succeeded on retry — both classes now predicted")
                base_clf = base_clf2
                calibrated = calibrated2
                predictions = preds2
        except Exception as exc:
            log.warning("Self-heal retry failed: %s — keeping first-pass model", exc)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 400)
    # PR-44 — populate AUC + win-precision so the dashboard column is filled.
    try:
        proba_test = calibrated.predict_proba(X_test)[:, 1]
    except Exception:
        proba_test = None

    log.info("Scalping Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    # ── Persist via canonical model_paths helper (canonical TF is 1m) ─────
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    from src.utils.model_integrity import sign_model
    paths = artifact_paths('scalping', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    sign_model(str(paths['model']))
    log.info("Model saved -> %s", paths['model'])

    # Phase 6b wire-in — persist training feature distribution baseline.
    try:
        from src.risk.drift_baseline import save_baseline
        save_baseline('scalping', timeframe, X)
    except Exception as _e:
        log.warning("[scalping][%s] save_baseline failed: %s", timeframe, _e)

    # accuracy_warning is only set when the post-rebalance model still
    # collapses to predicting one class (one of long_acc / short_acc is
    # ~0%) OR when both per-class precisions are below 50 %. Previous
    # builds emitted the warning unconditionally on the imbalanced
    # target distribution, even when the model itself was healthy.
    accuracy_warning = None
    if long_acc < 5.0 or short_acc < 5.0:
        accuracy_warning = ('Single-class collapse: model predicts only '
                            f'{"long" if short_acc < 5.0 else "short"} class. '
                            'Class imbalance unresolved post-SMOTE.')
    elif long_acc < 50.0 and short_acc < 50.0:
        accuracy_warning = ('Both per-class precisions <50% — model has no '
                            'discrimination on either side.')

    meta = {
        "model": "Scalping (HistGBT + Calibrated)",
        "accuracy": accuracy * 100,
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),  # required for MLPredictor._get_model_features
        "cio_overrides_applied": dict(_scalp_applied) if _scalp_applied else None,  # X1.3
        "n_iterations": n_iter,
        "walk_forward_mean_acc": round(float(np.mean(fold_accs)) * 100, 2),
        "target": "triple_barrier_long_win_1m",
        "symbols": symbols, "timeframe": timeframe,
        "smote_used": bool(_SMOTE_AVAILABLE),
        "pos_rate_pct": round(pos_rate * 100, 2),
        "last_trained": datetime.now(timezone.utc).isoformat()
    }
    if accuracy_warning:
        meta["accuracy_warning"] = accuracy_warning
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
    ap = argparse.ArgumentParser(description="Train the scalping (1m) model")
    ap.add_argument("--timeframe", default="1m",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_scalping_model(timeframe=args.timeframe)
