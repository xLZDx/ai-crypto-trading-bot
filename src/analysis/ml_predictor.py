"""
MLPredictor — universal inference wrapper for all sklearn/joblib models.

Key design: builds a rich, superset feature DataFrame, then reads the model's
own `feature_names_in_` to select exactly the columns it was trained on.
This avoids hardcoding feature lists that drift out of sync with training scripts.
"""
import io
import os
import sys
import logging
import traceback

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.safe_json import read_json
from src.utils.model_integrity import verify_and_load_bytes

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self, model_filename: str = "btc_rf_model.joblib", model_type: str = "base"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_dir, "models", model_filename)
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        self.accuracy = 0.0
        self.long_accuracy = 0.0
        self.short_accuracy = 0.0
        self.last_error = ""
        self.last_status = "init"          # 'ok'|'low_confidence'|'no_data'|'not_loaded'|'error'|'init'
        self._last_confidence = 0.5
        self.optimal_threshold = 0.52      # overridden from meta JSON after load
        self._symbols_sorted: list = []    # overridden from meta JSON after load
        self._embedded_features: list[str] | None = None
        # Locked feature list — resolved once at __init__ via the same
        # priority chain that _get_model_features() uses (embedded → meta
        # JSON → recursive find_features → hardcoded). Frozen here so a
        # post-init meta JSON rewrite (e.g. another process retrains the
        # model and bumps n_features) cannot make us pass N+k columns to
        # an in-memory model trained on N. Was bug 2026-05-14: bot loaded
        # 20-feat trend_model.joblib, then retrain rewrote meta JSON to
        # 22 features; every predict re-read meta → 22 cols → XGBoost
        # raised "Feature shape mismatch, expected: 20, got 22" ×2315.
        self._features: list[str] | None = None

        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(io.BytesIO(verify_and_load_bytes(self.model_path)))
                # train_model_v2.py wraps as {"model": estimator, "feature_cols": [...]};
                # legacy trainers dump the estimator directly. Accept both.
                if isinstance(loaded, dict) and "model" in loaded and hasattr(loaded["model"], "predict"):
                    self.model = loaded["model"]
                    cols = loaded.get("feature_cols") or loaded.get("features")
                    if isinstance(cols, list) and cols:
                        self._embedded_features = list(cols)
                else:
                    self.model = loaded
                self.is_loaded = True
                logger.info("ML Model loaded: %s", self.model_path)
                meta_path = self.model_path.replace(".joblib", "_meta.json")
                meta = read_json(meta_path, default={})
                self.accuracy          = meta.get("accuracy", 0.0)
                self.long_accuracy     = meta.get("long_accuracy", 0.0)
                self.short_accuracy    = meta.get("short_accuracy", 0.0)
                self.optimal_threshold  = float(meta.get("optimal_threshold", 0.52))
                self._symbols_sorted    = meta.get("symbols_sorted", [])
                # Resolve and freeze the feature list NOW so it stays in
                # lockstep with the in-memory model object.
                self._features = self._resolve_features()
                if self._features:
                    logger.info(
                        "MLPredictor[%s] locked features at init: %d cols",
                        os.path.basename(self.model_path), len(self._features),
                    )
            except Exception as exc:
                logger.error("Failed to load ML model from %s: %s", self.model_path, exc)
                self.last_error = f"Model load failed: {exc}"
        else:
            logger.warning("ML Model not found: %s", self.model_path)
            self.last_error = f"Model file not found: {self.model_path}"

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, data, symbol: str = "") -> int | None:
        """
        Return 1 (bullish), 0 (bearish), or None (no signal / low confidence).
        Accepts a list of OHLCV dicts or a DataFrame.

        Sets `self.last_status` to one of:
            'ok'             — signal returned (1 or 0)
            'low_confidence' — model is below 0.52 threshold (None returned)
            'no_data'        — fewer than 30 candles available
            'not_loaded'     — model file missing
            'error'          — exception raised inside predict()
        `self.last_error` is populated ONLY on real exceptions ('error') or
        'not_loaded'. Callers that previously checked `last_error` to detect
        problems should now check `last_status == 'error'` to avoid mis-
        labelling normal low-confidence outcomes.
        """
        self.last_error = ""
        self.last_status = "ok"
        if not self.is_loaded or self.model is None:
            self.last_status = "not_loaded"
            self.last_error = "Model not loaded"
            return None
        if len(data) < 30:
            self.last_status = "no_data"
            return None

        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = self._build_all_features(df, symbol=symbol)
            features = self._get_model_features()

            # Ensure every expected column exists (fill unknown with 0)
            for f in features:
                if f not in df.columns:
                    df[f] = 0.0

            last_row = df.iloc[-1:][features].fillna(0)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(last_row)[0]
                p_long = float(proba[1]) if len(proba) > 1 else float(proba[0])
                self._last_confidence = p_long
                if p_long >= self.optimal_threshold:
                    return 1
                elif (1.0 - p_long) >= self.optimal_threshold:
                    return 0
                else:
                    # Low confidence is a normal, expected outcome — do NOT
                    # set last_error (that's reserved for actual exceptions).
                    self.last_status = "low_confidence"
                    return None
            else:
                result = int(self.model.predict(last_row)[0])
                self._last_confidence = 0.55
                if result not in (0, 1):
                    self.last_status = "error"
                    self.last_error = f"Unexpected prediction value: {result}"
                    return None
                return result

        except Exception as exc:
            self.last_status = "error"
            self.last_error = f"ML Prediction Error: {exc}"
            logger.error(self.last_error)
            logger.debug("ML Prediction Error traceback:\n%s", traceback.format_exc())
            return None

    def predict_proba_long(self, data) -> float:
        """Return P(long win) in [0, 1]. Used by Kelly sizer and RiskAgent."""
        self.predict(data)
        return getattr(self, "_last_confidence", 0.5)

    # ── Feature building ──────────────────────────────────────────────────────

    def _get_model_features(self) -> list[str]:
        """
        Return the cached feature list (resolved once at __init__).
        Frozen on init so meta-JSON rewrites by a concurrent trainer cannot
        drift the runtime feature count away from the in-memory model.
        """
        if self._features is not None:
            return self._features
        # Fallback for callers that constructed MLPredictor with a missing
        # model file (is_loaded=False) — still resolve so feature consumers
        # (meta_labeler training, etc.) get a usable list.
        return self._resolve_features()

    def _resolve_features(self) -> list[str]:
        """
        Resolve the feature list this predictor will pass to model.predict().
        Authoritative source: the IN-MEMORY model's `n_features_in_` count
        (walked recursively through CalibratedClassifierCV + the XGB wrapper
        in src/utils/gpu_classifier.py:_clf). If the count is known, every
        candidate list is filtered to only those whose len matches — that
        way a stale meta JSON or out-of-sync hardcoded list can never
        produce a shape mismatch against the loaded joblib.

        Priority chain (each candidate must satisfy any count constraint):
          0. _embedded_features (joblib payload dict, train_model_v2 path)
          1. meta JSON sibling (<model>_meta.json features list)
          2. recursive feature_names_in_ on the model
          3. type-specific hardcoded list

        Why this matters: 2026-05-14 hit a case where the bot loaded
        trend_model.joblib (22-feat XGB) but a maintenance job had clobbered
        trend_model_meta.json to {n_features:20, features:[]}. The meta
        skipped because features.len==0, find_features failed because the
        XGB wrapper hides feature_names_in_, hardcoded returned 20 — but
        the model expected 22. Now we walk the model FIRST to learn its
        expected count, and reject any candidate list whose length doesn't
        match. If nothing matches, synthesize fX columns of the right
        length so the predict path can't shape-mismatch.
        """
        expected_n = self._inspect_model_n_features()

        def _accept(lst: list[str] | None) -> bool:
            if not lst:
                return False
            if expected_n is None:
                return True
            return len(lst) == expected_n

        # 0. Embedded features from the joblib payload.
        if _accept(self._embedded_features):
            return list(self._embedded_features)

        # 1. Sibling meta JSON. Two naming conventions exist:
        #    a. Canonical: `btc_rf_model.joblib` -> `btc_rf_model_meta.json`
        #    b. Per-TF:    `base_15m_model.joblib` -> `base_15m_meta.json`
        # Phase K.3 (2026-05-14): try both forms so per-TF predictors can
        # read meta JSON instead of falling all the way through to the
        # hardcoded list (which lost feature_names + n_features info).
        candidates: list[str] = []
        a = self.model_path.replace(".joblib", "_meta.json")
        candidates.append(a)
        # Per-TF convention: drop the "_model" infix.
        if "_model.joblib" in self.model_path:
            b = self.model_path.replace("_model.joblib", "_meta.json")
            candidates.append(b)
        for meta_path in candidates:
            if not os.path.exists(meta_path):
                continue
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # Phase L follow-up (2026-05-14) — accept the three keys
                # different trainers use for their feature list:
                #   features      — train_model / train_trend / train_futures / train_scalping
                #   feature_names — older legacy schemas
                #   meta_features — train_meta_labeler.py (line 439)
                # Without `meta_features` here, MLPredictor falls through
                # to the recursive search on meta_labeler.joblib and ends
                # up returning the wrong list.
                for key in ("features", "feature_names", "meta_features"):
                    cand = meta.get(key) if isinstance(meta.get(key), list) else None
                    if _accept(cand):
                        return list(cand)
            except Exception as e:
                logger.debug("Could not read features from meta json %s: %s", meta_path, e)

        # 2. Recursive feature_names_in_ probe.
        def find_features(obj, depth=0):
            if depth > 5 or obj is None:
                return None
            if hasattr(obj, "feature_names_in_"):
                fns = getattr(obj, "feature_names_in_", None)
                if fns is not None:
                    try:
                        return list(fns)
                    except Exception:
                        return None
            for attr in ["_clf", "estimator", "base_estimator", "best_estimator_",
                         "model", "_final_estimator", "step"]:
                if hasattr(obj, attr):
                    res = find_features(getattr(obj, attr), depth + 1)
                    if res:
                        return res
            if hasattr(obj, "calibrated_classifiers_"):
                for clf in getattr(obj, "calibrated_classifiers_"):
                    res = find_features(clf, depth + 1)
                    if res:
                        return res
            if hasattr(obj, "steps"):
                for name, step in getattr(obj, "steps"):
                    res = find_features(step, depth + 1)
                    if res:
                        return res
            return None

        recursed = find_features(self.model)
        if _accept(recursed):
            return list(recursed)

        # 3. Hardcoded fallback list.
        hardcoded = self._hardcoded_features()
        if _accept(hardcoded):
            return list(hardcoded)

        # 4. Last-ditch: if the in-memory model says it wants N features
        # but no candidate list matched, synthesize fX names that match the
        # count. The predict path fills unknown columns with 0, so this is
        # safe — better than letting a shape mismatch propagate.
        if expected_n is not None:
            logger.warning(
                "[MLPredictor] %s: no feature list matched model's "
                "n_features_in_=%d (embedded=%s, meta=%s, recursed=%s, "
                "hardcoded=%s). Synthesizing fX placeholders.",
                os.path.basename(self.model_path), expected_n,
                len(self._embedded_features) if self._embedded_features else None,
                len(meta.get("features") or []) if 'meta' in locals() and isinstance(meta, dict) else None,
                len(recursed) if recursed else None,
                len(hardcoded) if hardcoded else None,
            )
            return [f"f{i}" for i in range(expected_n)]

        logger.debug("[MLPredictor] feature_names_in_ not found -- using hardcoded list for '%s'", self.model_type)
        return hardcoded or []

    def _inspect_model_n_features(self) -> int | None:
        """Walk the loaded model to find its actual `n_features_in_`. This
        is the authoritative count the model will accept at predict time —
        meta JSON / hardcoded lists must match it.
        """
        if self.model is None:
            return None
        seen: set[int] = set()
        def walk(obj, depth=0):
            if depth > 6 or obj is None or id(obj) in seen:
                return None
            seen.add(id(obj))
            n = getattr(obj, "n_features_in_", None)
            if isinstance(n, int) and n > 0:
                return n
            for a in ("_clf", "estimator", "base_estimator", "best_estimator_",
                      "model", "_final_estimator"):
                if hasattr(obj, a):
                    r = walk(getattr(obj, a), depth + 1)
                    if r is not None:
                        return r
            if hasattr(obj, "calibrated_classifiers_"):
                for c in obj.calibrated_classifiers_:
                    r = walk(c, depth + 1)
                    if r is not None:
                        return r
            if hasattr(obj, "steps"):
                for _, step in obj.steps:
                    r = walk(step, depth + 1)
                    if r is not None:
                        return r
            # XGBoost wrapped path — inner xgboost.XGBClassifier has its
            # own n_features_in_ but the wrapper hides it.
            if hasattr(obj, "get_booster"):
                try:
                    b = obj.get_booster()
                    nf = b.num_features()
                    if nf:
                        return int(nf)
                except Exception:
                    pass
            return None
        return walk(self.model)

    def _hardcoded_features(self) -> list[str]:
        """Type-specific fallback feature lists matching each training script.

        IMPORTANT: keep in sync with the trainer's FEATURE_COLUMNS / FEATURES
        lists. Previous list lost track of 4 features
        (trend_strength, vol_regime, is_trending, is_volatile) when
        train_scalping_model.py grew them — runtime then built 17-column
        DataFrames against 21-feature models → "expected 21, got 17" loops.
        """
        if self.model_type == "scalping":
            # Mirrors src/engine/train_scalping_model.py:FEATURE_COLUMNS (21 entries).
            return [
                "frac_diff_d40", "rsi_7", "macd_fast", "volume_surge",
                "dist_to_micro_supp", "taker_buy_ratio", "avg_trade_size",
                "hour", "roc_3", "roc_5", "roc_10", "bb_pb",
                "ofi_z", "vwap_dist", "kc_pos", "signal_rsi", "signal_bb",
                "trend_strength", "vol_regime", "is_trending", "is_volatile",
            ]
        if self.model_type == "futures":
            return [
                "return", "rsi_14", "dist_to_support", "volume_drop",
                "hour", "roc_5", "funding_z", "funding_positive",
                "funding_negative", "dist_to_supply", "dist_to_demand",
                "liq_proximity", "frac_diff_d40",
            ]
        if self.model_type == "trend":
            # Mirrors src/engine/train_trend_model.py:FEATURE_COLUMNS (20 entries).
            return [
                "frac_diff_d40", "macd", "macd_signal", "macd_hist",
                "trend_alignment", "volume_surge", "atr_14", "adx_14",
                "don_pos_20", "kc_pos", "kc_width",
                "vwap_dist", "ofi_z", "funding_z",
                "signal_macd", "signal_don",
                "trend_strength", "vol_regime", "is_trending", "is_volatile",
            ]
        # base — mirrors src/engine/train_model.py:FEATURE_COLUMNS (33 entries).
        return [
            "frac_diff_d40", "volatility", "dist_sma_7", "dist_sma_30",
            "rsi_14", "macd", "macd_hist", "volume_momentum", "stoch_k",
            "return_lag1", "return_lag2", "return_lag3", "return_lag5",
            "atr_pct", "taker_buy_ratio", "avg_trade_size",
            "hour", "day_of_week", "roc_14", "roc_3", "roc_7",
            "bb_pb", "news_sentiment", "ofi_z", "vwap_dist", "liq_proximity",
            "trend_strength", "vol_regime", "is_trending", "is_volatile",
            "signal_rsi", "signal_macd", "signal_bb",
        ]

    def _build_all_features(
        self, df: pd.DataFrame, symbol: str = "", timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Build every feature that any model type might need — a full superset.
        Each indicator is wrapped individually so one failure doesn't abort prediction.
        Pass symbol/timeframe to also merge CoinGlass v4 features (macro always
        available; per-symbol futures features require the futures download to
        have run).  If symbol is empty, only macro features are merged.
        """
        from src.analysis.feature_engineering import (
            add_taker_and_trade_features, add_rsi, add_macd,
            add_bollinger_bands, add_roc, add_time_features,
            add_adx, add_atr, add_ofi, add_vwap,
            add_donchian, add_keltner, add_funding_zscore,
            add_liquidity_proximity,
        )
        from src.analysis.fractional_diff import add_fractional_diff

        def _safe(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.debug("[MLPredictor] indicator %s failed: %s", fn.__name__, exc)
                return args[0] if args else df

        # ── Core returns & time ───────────────────────────────────────────────
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        if "timestamp" in df.columns:
            df = _safe(add_time_features, df)
        else:
            df["hour"] = 12
            df["day_of_week"] = 0

        df = _safe(add_taker_and_trade_features, df)

        # ── RSI ───────────────────────────────────────────────────────────────
        df = _safe(add_rsi, df, period=7,  col_name="rsi_7")
        df = _safe(add_rsi, df, period=14, col_name="rsi_14")

        # ── MACD ─────────────────────────────────────────────────────────────
        df = _safe(add_macd, df)
        try:
            _fast = add_macd(df.copy(), fast=5, slow=13, signal=3, prefix="")
            df["macd_fast"] = _fast["macd"]
        except Exception:
            df["macd_fast"] = 0.0

        # ── Bollinger Bands ───────────────────────────────────────────────────
        df = _safe(add_bollinger_bands, df, window=20)

        # ── ROC ───────────────────────────────────────────────────────────────
        df = _safe(add_roc, df, [3, 5, 7, 10, 14])

        # ── ATR / ADX ─────────────────────────────────────────────────────────
        df = _safe(add_atr, df, period=14)
        df = _safe(add_adx, df, period=14)

        # ── OFI, VWAP, Donchian, Keltner ─────────────────────────────────────
        df = _safe(add_ofi, df, window=20)
        df = _safe(add_vwap, df)
        df = _safe(add_donchian, df, n=20)
        df = _safe(add_keltner, df)

        # ── Funding Z-score ───────────────────────────────────────────────────
        df = _safe(add_funding_zscore, df)

        # ── Liquidity proximity ───────────────────────────────────────────────
        df = _safe(add_liquidity_proximity, df)

        # ── Fractional differentiation ────────────────────────────────────────
        try:
            df = add_fractional_diff(df, d=0.4)
        except Exception:
            df["frac_diff_d40"] = 0.0

        # ── Derived stats ─────────────────────────────────────────────────────
        df["volatility"]  = df["return"].rolling(7, min_periods=2).std()
        df["sma_7"]       = df["close"].rolling(7).mean()
        df["sma_30"]      = df["close"].rolling(30).mean()
        df["sma_50"]      = df["close"].rolling(50).mean()
        df["sma_200"]     = df["close"].rolling(200).mean()
        df["dist_sma_7"]  = df["close"] / df["sma_7"].replace(0, np.nan) - 1
        df["dist_sma_30"] = df["close"] / df["sma_30"].replace(0, np.nan) - 1
        df["trend_alignment"] = (df["sma_50"] > df["sma_200"]).astype(float)

        vol_sma_5  = df["volume"].rolling(5).mean().replace(0, np.nan)
        vol_sma_7  = df["volume"].rolling(7).mean().replace(0, np.nan)
        vol_sma_14 = df["volume"].rolling(14).mean().replace(0, np.nan)
        vol_sma_20 = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["volume_momentum"] = df["volume"] / vol_sma_14
        df["volume_surge"]    = (df["volume"] > vol_sma_20 * 1.5).astype(float)
        df["volume_drop"]     = (df["volume"] < vol_sma_7 * 0.7).astype(float)

        high_14 = df["high"].rolling(14).max()
        low_14  = df["low"].rolling(14).min()
        df["stoch_k"] = (df["close"] - low_14) / (high_14 - low_14).replace(0, np.nan) * 100

        for lag in (1, 2, 3, 5):
            df[f"return_lag{lag}"] = df["return"].shift(lag)
            df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)

        df["atr_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Scalping-specific
        df["vol_sma_5"]        = vol_sma_5
        df["low_15"]           = df["low"].rolling(15).min()
        df["dist_to_micro_supp"] = (df["close"] - df["low_15"]) / df["close"].replace(0, np.nan)

        # Futures-specific
        df["low_30"]           = df["low"].rolling(30).min()
        df["dist_to_support"]  = (df["close"] - df["low_30"]) / df["close"].replace(0, np.nan)

        # Regime Features
        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        df['trend_strength'] = (df['ema_fast'] - df['ema_slow']).abs() / df['atr_14'].replace(0, 1e-9)
        df['vol_short'] = df['return'].rolling(window=7).std()
        df['vol_long'] = df['return'].rolling(window=100, min_periods=10).std()
        df['vol_regime'] = df['vol_short'] / df['vol_long'].replace(0, 1e-9)
        df['is_trending'] = (df['trend_strength'] > 1.5).astype(int)
        df['is_volatile'] = (df['vol_regime'] > 1.5).astype(int)

        # Strategy signals used as ML features in some models
        rsi_s  = df.get("rsi_14", pd.Series(50.0, index=df.index))
        macd_h = df.get("macd_hist", pd.Series(0.0, index=df.index))
        bb_p   = df.get("bb_pb", pd.Series(0.5, index=df.index))
        df["signal_rsi"]  = np.where(rsi_s < 30, 1.0, np.where(rsi_s > 70, -1.0, 0.0))
        df["signal_macd"] = np.where(macd_h > 0, 1.0, -1.0)
        df["signal_bb"]   = np.where(bb_p < 0.1, 1.0, np.where(bb_p > 0.9, -1.0, 0.0))

        df["news_sentiment"] = 0.0

        # Explicit regime one-hot features
        try:
            from src.analysis.feature_engineering import add_explicit_regime
            df = add_explicit_regime(df)
        except Exception as _reg_exc:
            logger.debug("[MLPredictor] add_explicit_regime failed: %s", _reg_exc)
            for _rc in ('regime_bull', 'regime_bear', 'regime_chop', 'regime_high_vol'):
                if _rc not in df.columns:
                    df[_rc] = 0.0

        # symbol_id — ordinal encoding (1-based); 0 = unknown/not in training set
        try:
            if symbol and self._symbols_sorted:
                df['symbol_id'] = float(self._symbols_sorted.index(symbol) + 1)
            else:
                df['symbol_id'] = 0.0
        except (ValueError, AttributeError):
            df['symbol_id'] = 0.0

        # CoinGlass v4 enrichment — macro always available (fear_greed,
        # btc_dominance, stablecoin_mcap); per-symbol futures features only
        # when symbol is provided and the futures download has run.
        try:
            from src.analysis.feature_engineering import add_coinglass_features
            df = add_coinglass_features(df, symbol, timeframe)
        except Exception as _cg_exc:
            logger.debug("[MLPredictor] CoinGlass features skipped: %s", _cg_exc)

        return df
