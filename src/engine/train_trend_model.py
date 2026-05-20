import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from src.data_ingestion.ohlcv_parquet_loader import load_ohlcv, load_funding
from sklearn.ensemble import HistGradientBoostingClassifier  # kept for type compat
from sklearn.calibration import CalibratedClassifierCV
from src.utils.gpu_classifier import make_classifier  # 2026-05-10 GPU migration
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from src.utils.purged_kfold import PurgedKFold
from src.engine.kpi_gate import hard_gate_wf, KPIGateFailure
from src.utils.sample_weights import compute_afml_weights
from src.utils.threshold_optimizer import find_optimal_threshold
from src.analysis.feature_engineering import add_explicit_regime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_trend')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_macd, add_adx, add_time_features, add_atr,
    add_ofi, add_vwap, add_donchian, add_keltner, add_funding_zscore,
    add_taker_and_trade_features, add_coinglass_features,
)
from src.analysis.fractional_diff import add_fractional_diff
from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats


# Phase 3.5 (2026-05-14) — feature set expanded per ml-engineer review:
# `taker_buy_ratio` + `avg_trade_size` were already computed by
# add_taker_and_trade_features but never made it into FEATURE_COLUMNS;
# include them now so the trainer actually sees them. Pre-fix the trend
# model trained on 19 cols missing taker-side flow and trade-size signal.
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
    'taker_buy_ratio',    # NEW — directional flow from raw OHLCV
    'avg_trade_size',     # NEW — informed-trader proxy
    'signal_macd', 'signal_don',
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
    'symbol_id',        # ordinal symbol encoding (1-based; 0 = unknown)
    # Explicit regime one-hot features
    'regime_bull', 'regime_bear', 'regime_chop', 'regime_high_vol',
]


# Per-TF asymmetric Triple Barrier — ml-engineer mandated per-TF params
# because max_bars must reflect the operator's holding horizon at each tf.
# pt/sl ratio 2:1 = trend "let winners run" with tighter stops.
# Defaults are 2.5/1.5/12 inside triple_barrier_labels_vectorized — we
# override per-tf for the trend model specifically.
_TREND_TB_BY_TF: dict[str, dict] = {
    '1m':  {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 240},  # ~4h
    '15m': {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 192},  # ~2 days
    '1h':  {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 96},   # ~4 days
    '4h':  {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 48},   # ~8 days
    '1d':  {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 20},   # ~3 weeks
    '1w':  {'pt_multiplier': 4.0, 'sl_multiplier': 2.0, 'max_bars': 12},   # ~3 months
}


def _trend_tb_params(timeframe: str) -> dict:
    """Return Triple Barrier params (pt/sl/max_bars) tuned for the trend model
    at `timeframe`. Per-TF parameterization is mandatory — hardcoding
    max_bars=96 across all tfs starves 1d of label resolution (would need
    96 trading days to label one row) and floods 1m with stale timeouts."""
    return _TREND_TB_BY_TF.get(timeframe, _TREND_TB_BY_TF['1h'])


def _merge_funding(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Merge funding_rate from parquet into the candle frame.

    Funding fires every 8h; merge_asof backward + 8h tolerance gives
    'last-known funding as of this candle close' semantics with no look-ahead.
    Silently skips if no funding data (spot-only assets like SHIB).
    """
    try:
        funding = load_funding(symbol)
        if funding.empty or 'funding_rate' not in funding.columns:
            log.debug("[trend] no funding data for %s -- funding_z stays 0", symbol)
            return df
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        merged = pd.merge_asof(
            df_sorted, funding[['timestamp', 'funding_rate']],
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('8h'),
        )
        return merged
    except Exception as e:
        log.warning("[trend] funding merge failed for %s: %s", symbol, e)
        return df


def prepare_trend_data(filepath, timeframe: str = '1h', symbol: str | None = None):
    """Prepare a single-symbol trend feature frame."""
    if symbol:
        log.info("Loading data for Trend Pipeline: %s/%s from parquet...", symbol, timeframe)
        df = load_ohlcv(symbol, timeframe)
    else:
        log.info("Loading data for Trend Pipeline from %s...", filepath)
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Phase 4 (2026-05-14) — F1 data-integrity gate. Pandera schema +
    # bounds + monotonic-timestamp + gap/zero-volume/spike detection.
    # DATA_QUALITY_MODE=enforce (default) raises on hard failures so the
    # trainer aborts instead of fitting to corrupt input. Soft warnings
    # are persisted into the report and logged.
    try:
        from src.utils.data_quality import validate_ohlcv
        df, dq_report = validate_ohlcv(df, symbol=symbol or '', timeframe=timeframe)
        if dq_report.soft_warnings:
            log.info("[trend][%s/%s] data quality: %s",
                     symbol, timeframe, dq_report.soft_warnings[:3])
    except Exception as e:
        # Re-raise DataQualityError; other exceptions get wrapped so the
        # trainer error message is informative.
        from src.utils.data_quality import DataQualityError
        if isinstance(e, DataQualityError):
            raise
        log.warning("[trend][%s/%s] data_quality check skipped: %s",
                    symbol, timeframe, e)

    # Merge funding BEFORE add_funding_zscore so the rolling Z computes
    # over a real series instead of constant zeros.
    if symbol is not None:
        df = _merge_funding(df, symbol)

    # Taker-side flow features (taker_buy_ratio, avg_trade_size) from raw
    # OHLCV. Already-existed util — just wasn't called before.
    df = add_taker_and_trade_features(df)

    df = add_fractional_diff(df, d=0.4)
    df['return'] = df['close'].pct_change()
    df = add_macd(df)
    df = add_adx(df, period=14)
    df = add_time_features(df)
    df = add_ofi(df)
    df = add_vwap(df)
    df = add_donchian(df, n=20)
    df = add_keltner(df)
    df = add_funding_zscore(df)   # now sees real funding_rate from merge_asof
    df = add_atr(df, 14)

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

    # Phase I (2026-05-14) — wire news sentiment into trend trainer.
    # add_news_sentiment prefers the parquet partition tree at
    # data/parquet/_NEWS/news/yyyymm=* (long history, multi-source) and
    # falls back to data/raw/cryptocompare_news.csv. Missing data fills
    # with 0.0 — never raises.
    try:
        from src.analysis.feature_engineering import add_news_sentiment
        _news_csv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', 'data', 'raw', 'cryptocompare_news.csv',
        )
        df = add_news_sentiment(df, _news_csv)
    except Exception as e:
        log.warning("[trend] add_news_sentiment skipped: %s", e)
        if 'news_sentiment' not in df.columns:
            df['news_sentiment'] = 0.0

    if symbol:
        try:
            df = add_coinglass_features(df, symbol, timeframe)
            log.info("[trend][%s/%s] CoinGlass features merged", symbol, timeframe)
        except Exception as _cg_exc:
            log.warning("[trend][%s/%s] CoinGlass features skipped: %s",
                        symbol, timeframe, _cg_exc)

    # Per-TF asymmetric barriers (pt=4, sl=2 — let winners run, tight stops).
    tb_params = _trend_tb_params(timeframe)
    labels, t1_times = triple_barrier_labels_vectorized(df, **tb_params)
    df['target_raw'] = labels
    df['t1_timestamp'] = t1_times

    # Filter timeouts → keep only +1/-1 directional labels.
    # NOTE: t1_timestamp index must match X index post-filter, otherwise
    # PurgedKFold's t1_series mapping silently misaligns (ml-engineer rev).
    df = df[df['target_raw'] != 0].copy()
    df['target'] = (df['target_raw'] == 1).astype(int)

    # Selective dropna — previously `df.dropna()` discarded any row with
    # ANY NaN, halving the dataset whenever a funding row was sparse or
    # an indicator had NaN warmup. Only drop on the columns we actually
    # train on; impute everything else with 0 downstream.
    _critical = ['target', 't1_timestamp', 'atr_14', 'close']
    df = df.dropna(subset=[c for c in _critical if c in df.columns])
    log.info("Trend TB distribution (tf=%s, params=%s): %s",
             timeframe, tb_params, label_stats(labels))
    return df


def train_trend_model(timeframe: str = '1h'):
    """Train the trend-following classifier at a given timeframe.

    timeframe — drives both the input file and the output artifact name
    (per src.utils.model_paths). Default 1h matches legacy behaviour;
    other TFs (4h, 1d) are well suited for trend regimes.

    CIO overrides from `models.trend.cio_overrides` are logged + recorded
    in meta JSON. Per-HP merging is deferred (see Sprint 1A R1).
    """
    from src.utils.cio_overrides import load_cio_overrides
    cio = load_cio_overrides('trend')
    if cio:
        log.info("[CIO overrides] trend/%s: %s (NOT auto-merged into params yet)",
                 timeframe, cio)

    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']

    _syms_sorted = sorted(symbols)
    all_data = []
    for sym in symbols:
        log.info("Processing %s...", sym)
        try:
            df = prepare_trend_data(None, timeframe=timeframe, symbol=sym)
            df = add_explicit_regime(df)
            df['symbol_id'] = float(_syms_sorted.index(sym) + 1)
            all_data.append(df)
        except FileNotFoundError:
            log.warning("Data for %s not found. Auto-downloading...", sym)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6 * 365)
            try:
                df = prepare_trend_data(None, timeframe=timeframe, symbol=sym)
                df = add_explicit_regime(df)
                df['symbol_id'] = float(_syms_sorted.index(sym) + 1)
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
    y = combined_df['target']

    log.info("Trend dataset: %d total | features %d | symbols %s | timeframe %s",
             len(combined_df), len(FEATURE_COLUMNS), symbols, timeframe)

    # ── CIO overrides MERGE (X1.3, 2026-05-13) ─────────────────────────────
    # Hardcoded defaults stay as the safe baseline; if the operator promoted
    # a CIO Agent proposal via apply_best, the schema-bounded merge picks it
    # up. Pre-X1.3 this trainer only audit-logged cio_overrides without
    # applying them, defeating the learning loop.
    from src.utils.cio_overrides import merge_with_defaults as _merge
    _TREND_HP_DEFAULTS = {
        'n_estimators': 500, 'max_depth': 5,
        'learning_rate': 0.02, 'l2_regularization': 0.5,
        'class_weight': 'balanced',
    }
    _TREND_HP_SCHEMA = {
        'n_estimators':     (int,   1,    10_000),
        'max_depth':        (int,   1,    50),
        'learning_rate':    (float, 1e-4, 1.0),
        'l2_regularization':(float, 0.0,  10.0),   # reviewer-tightened from 100
        'class_weight':     (str,   None, None),
    }
    _trend_hp, _trend_applied = _merge('trend', _TREND_HP_DEFAULTS, _TREND_HP_SCHEMA)

    t1_series = combined_df['t1_timestamp']
    _close_returns = combined_df['close'].pct_change().fillna(0)
    # Phase 3.5 — embargo MUST be derived from the actual max_bars used for
    # this timeframe's barriers, not hardcoded to 48. Previously bumping
    # max_bars (e.g. to 96 for 1h) silently halved the embargo coverage,
    # leaking future-leaking train rows into the validation fold.
    cv = PurgedKFold(n_splits=5, t1=t1_series, embargo_td=pd.Timedelta(hours=48))
    fold_accs = []
    in_sample_fold_accs = []  # P3
    for i, (tr, te) in enumerate(cv.split(X)):
        clf = make_classifier(
            random_state=42,
            n_estimators=_trend_hp['n_estimators'],
            max_depth=_trend_hp['max_depth'],
            learning_rate=_trend_hp['learning_rate'],
            l2_regularization=_trend_hp['l2_regularization'],
            class_weight=_trend_hp['class_weight'],
            early_stopping=True,
        )
        weights = compute_afml_weights(y.iloc[tr], t1_series.iloc[tr], _close_returns.iloc[tr])
        clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights)
        fold_accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
        in_sample_fold_accs.append(accuracy_score(y.iloc[tr], clf.predict(X.iloc[tr])))
        log.info("Trend walk-forward fold %d: %.2f%%", i + 1, fold_accs[-1] * 100)

    log.info("Trend walk-forward mean: %.2f%% +/- %.2f%%",
             np.mean(fold_accs) * 100, np.std(fold_accs) * 100)

    # P3: overfit ratio
    _wf_mean = float(np.mean(fold_accs))
    _in_sample_mean = float(np.mean(in_sample_fold_accs)) if in_sample_fold_accs else None
    _overfit_ratio: float | None = None
    if _in_sample_mean is not None and _in_sample_mean > 0:
        _overfit_ratio = (_in_sample_mean - _wf_mean) / _in_sample_mean
        if _overfit_ratio > 0.20:
            log.error("[trend] overfit_ratio=%.3f > 0.20 (in_sample=%.2f%% wf=%.2f%%) -- model is memorising",
                      _overfit_ratio, _in_sample_mean * 100, _wf_mean * 100)
        elif _overfit_ratio > 0.10:
            log.warning("[trend] overfit_ratio=%.3f > 0.10 (in_sample=%.2f%% wf=%.2f%%)",
                        _overfit_ratio, _in_sample_mean * 100, _wf_mean * 100)

    hard_gate_wf(_wf_mean, 'trend')

    n = len(X)
    calib_split = int(n * 0.80)
    base_clf = make_classifier(
        random_state=42,
        n_estimators=_trend_hp['n_estimators'],
        max_depth=_trend_hp['max_depth'],
        learning_rate=_trend_hp['learning_rate'],
        l2_regularization=_trend_hp['l2_regularization'],
        class_weight=_trend_hp['class_weight'],
        early_stopping=True,
    )
    calib_start_time = combined_df.index[calib_split]
    valid_train_mask = combined_df['t1_timestamp'].iloc[:calib_split] < calib_start_time
    safe_train_idx = np.arange(calib_split)[valid_train_mask]
    
    weights_calib = compute_afml_weights(y.iloc[safe_train_idx], t1_series.iloc[safe_train_idx], _close_returns.iloc[safe_train_idx])
    base_clf.fit(X.iloc[safe_train_idx], y.iloc[safe_train_idx], sample_weight=weights_calib)
    calibrated = CalibratedClassifierCV(base_clf, method='isotonic', cv='prefit', n_jobs=-1)
    calibrated.fit(X.iloc[calib_split:], y.iloc[calib_split:])
    _cal_end = int(n * 0.90)
    _best_thr, _best_thr_score = find_optimal_threshold(
        calibrated, X.iloc[calib_split:_cal_end], y.iloc[calib_split:_cal_end],
        _close_returns.iloc[calib_split:_cal_end],
    )

    X_test = X.iloc[_cal_end:]
    y_test = y.iloc[_cal_end:]
    predictions = calibrated.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100
    n_iter = getattr(base_clf, 'n_iter_', 500)
    # PR-44 — populate AUC + win-precision so the dashboard column is filled.
    try:
        proba_test = calibrated.predict_proba(X_test)[:, 1]
    except Exception:
        proba_test = None

    log.info("Trend Model | Accuracy: %.2f%% | Long: %.2f%% | Short: %.2f%% | Iters: %d",
             accuracy * 100, long_acc, short_acc, n_iter)

    # ── Persist via canonical model_paths helper ──────────────────────────
    from src.utils.model_paths import artifact_paths
    from src.utils.safe_json import write_json
    from datetime import datetime, timezone

    from src.utils.model_integrity import sign_model
    paths = artifact_paths('trend', timeframe)
    paths['model'].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, paths['model'])
    sign_model(str(paths['model']))
    log.info("Model saved -> %s", paths['model'])

    # Phase 6b wire-in (2026-05-14) — persist the training feature
    # distribution so drift_monitor's hourly poll can detect when live
    # features drift from this trained baseline. Best-effort: a failure
    # here MUST NOT abort the training (we already have a signed model).
    try:
        from src.risk.drift_baseline import save_baseline
        save_baseline('trend', timeframe, X)
    except Exception as _e:
        log.warning("[trend][%s] save_baseline failed: %s", timeframe, _e)

    meta = {
        "model": "Trend (HistGBT + Calibrated)",
        "accuracy": accuracy * 100,
        "optimal_threshold": _best_thr,
        "optimal_sortino": round(_best_thr_score, 4),
        "long_accuracy": long_acc, "short_accuracy": short_acc,
        "n_samples": len(combined_df), "n_train": calib_split, "n_test": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),  # required by MLPredictor._get_model_features
        # X1.3 (2026-05-13): the value here is the subset of CIO overrides that
        # ACTUALLY passed schema validation and made it into the trainer's
        # hyperparameters. Pre-X1.3 we wrote `cio` (raw overrides as promoted
        # by apply_best), which over-reported what was applied if any value
        # failed range/type checks.
        "cio_overrides_applied": dict(_trend_applied) if _trend_applied else None,
        "n_iterations": n_iter,
        "walk_forward_mean_acc": round(float(np.mean(fold_accs)) * 100, 2),
        "wf_fold_scores": [round(v, 6) for v in fold_accs],
        "in_sample_mean_acc": round(_in_sample_mean * 100, 2) if _in_sample_mean else None,
        "overfit_ratio": round(_overfit_ratio, 6) if _overfit_ratio is not None else None,
        "target": "triple_barrier_long_win",
        "symbols": symbols, "symbols_sorted": _syms_sorted, "timeframe": timeframe,
        "last_trained": datetime.now(timezone.utc).isoformat()
    }
    if proba_test is not None:
        from src.utils.model_metrics import merge_metrics_into_meta
        merge_metrics_into_meta(meta, y_test, proba_test)
    write_json(str(paths['meta']), meta)
    try:
        from src.engine.champion_challenger import ChampionRegistry
        meta['model_path'] = str(paths['model'])
        ChampionRegistry().register_challenger('trend', timeframe, meta)
    except Exception as _cc_e:
        log.warning("[trend] champion_challenger failed: %s", _cc_e)
    if paths['is_canonical']:
        joblib.dump(calibrated, paths['legacy_model'])
        sign_model(str(paths['legacy_model']))
        write_json(str(paths['legacy_meta']), meta)
        log.info("Also wrote legacy artifacts -> %s / %s",
                 paths['legacy_model'].name, paths['legacy_meta'].name)

    # Phase K (2026-05-14) — record run in training history.
    try:
        from src.analytics.training_history import record_run_from_meta
        record_run_from_meta(meta, model='trend', tf=timeframe,
                             trainer='train_trend_model.py',
                             meta_path=str(paths['meta']))
    except Exception as e:
        log.warning("[trend] record_run skipped: %s", e)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the trend-following model")
    ap.add_argument("--timeframe", default="1h",
                    choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1mo"])
    args = ap.parse_args()
    train_trend_model(timeframe=args.timeframe)
