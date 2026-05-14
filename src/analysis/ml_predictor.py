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
                self.accuracy       = meta.get("accuracy", 0.0)
                self.long_accuracy  = meta.get("long_accuracy", 0.0)
                self.short_accuracy = meta.get("short_accuracy", 0.0)
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

    def predict(self, data) -> int | None:
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

            df = self._build_all_features(df)
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
                if p_long >= 0.52:
                    return 1
                elif (1.0 - p_long) >= 0.52:
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
        Read the model's exact feature list from sklearn's feature_names_in_.
        Falls back to type-specific hardcoded lists if not available.
        Tries multiple attribute paths for CalibratedClassifierCV wrappers.
        """
        # 0. Trainers (e.g. train_model_v2) embed the feature list alongside
        # the estimator inside the joblib payload — prefer that over guessing.
        if self._embedded_features:
            return self._embedded_features

        # 1. Try to read from meta JSON
        meta_path = self.model_path.replace(".joblib", "_meta.json")
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "features" in meta and isinstance(meta["features"], list) and len(meta["features"]) > 0:
                    return meta["features"]
                if "feature_names" in meta and isinstance(meta["feature_names"], list) and len(meta["feature_names"]) > 0:
                    return meta["feature_names"]
            except Exception as e:
                logger.debug("Could not read features from meta json: %s", e)

        # 2. Try recursive search in the object
        def find_features(obj, depth=0):
            if depth > 5 or obj is None:
                return None
            if hasattr(obj, "feature_names_in_"):
                return list(obj.feature_names_in_)
            for attr in ["estimator", "base_estimator", "best_estimator_", "model", "_final_estimator", "step"]:
                if hasattr(obj, attr):
                    res = find_features(getattr(obj, attr), depth + 1)
                    if res: return res
            if hasattr(obj, "calibrated_classifiers_"):
                for clf in getattr(obj, "calibrated_classifiers_"):
                    res = find_features(clf, depth + 1)
                    if res: return res
            if hasattr(obj, "steps"):
                for name, step in getattr(obj, "steps"):
                    res = find_features(step, depth + 1)
                    if res: return res
            return None

        features = find_features(self.model)
        if features:
            return features

        logger.debug("[MLPredictor] feature_names_in_ not found — using hardcoded list for '%s'", self.model_type)
        return self._hardcoded_features()

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

    def _build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build every feature that any model type might need — a full superset.
        Each indicator is wrapped individually so one failure doesn't abort prediction.
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

        return df
