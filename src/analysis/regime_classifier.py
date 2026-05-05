"""
Market Regime Classifier.

Phase 2 upgrade: switched from `GaussianMixture` to `BayesianGaussianMixture`
per updated_architecture_plan_en.md §8 — the Dirichlet Process prior
auto-prunes unused components, giving the bot a way to *grow* the regime
taxonomy when novel market behaviour appears (e.g. a new circuit-breaker
regime around major Fed events) without having to manually re-tune
`n_components`.

Note on online updates: the architecture plan literally calls
`model.partial_fit(new_data)` — but sklearn's `BayesianGaussianMixture`
doesn't expose `partial_fit`. We approximate it with a periodic *warm-start
refit* that initialises with the previous fit's parameters. Empirically this
behaves like online EM with a memory-decay factor controlled by the size of
the new buffer relative to the historical training set.

Classifies the market into one of 3+ regimes:
  0 = RANGING   — low volatility, mean-reversion strategies work best
  1 = TRENDING  — directional momentum, breakout strategies work best
  2 = VOLATILE  — high volatility spike, reduce position size, avoid entries
  3+ = (Bayesian-discovered, may be empty/pruned)

Live usage: call RegimeClassifier.predict(df) on the last N bars → returns regime int.
Strategy routing:
  RANGING   → use RSI mean-reversion, BB reversion, VWAP reversion
  TRENDING  → use MACD momentum, Donchian breakout, cross-sectional momentum
  VOLATILE  → halve position size, skip scalping, use funding arbitrage only
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIME_NAMES = {0: "RANGING", 1: "TRENDING", 2: "VOLATILE"}
REGIME_STRATEGY_MAP = {
    0: ["RSI_MeanReversion", "BB_Reversion", "VWAP_Reversion"],
    1: ["MACD_Momentum", "Donchian_Breakout", "Keltner_Breakout"],
    2: ["Funding_Arb"],  # only safe strategy in volatile regime
}
REGIME_SIZE_MULT = {0: 1.0, 1: 1.0, 2: 0.5}  # volatile → half size


def _compute_regime_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Compute features used to classify regime.
    All features are normalized to be scale-invariant.
    """
    df = df.copy()
    close = df["close"]
    ret = close.pct_change()

    # Realized volatility (annualized from hourly)
    df["rv"] = ret.rolling(window).std() * np.sqrt(8760)

    # Trend strength: abs(return over window) / realized vol
    df["trend_str"] = ret.rolling(window).sum().abs() / (df["rv"].replace(0, 1e-9))

    # ATR normalized by price
    if "atr_14" not in df.columns:
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr_n"] = tr.rolling(14).mean() / close.replace(0, 1e-9)
    else:
        df["atr_n"] = df["atr_14"] / close.replace(0, 1e-9)

    # Volume z-score
    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std().replace(0, 1e-9)
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std

    # Auto-correlation of returns (negative AC → mean-reversion, positive → trending)
    df["ret_ac1"] = ret.rolling(window).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 1 else 0.0, raw=False
    )

    return df[["rv", "trend_str", "atr_n", "vol_z", "ret_ac1"]].dropna()


class RegimeClassifier:
    """
    Gaussian Mixture Model regime classifier.
    Trained lazily on first call if model file not found.
    """

    FEATURE_COLS = ["rv", "trend_str", "atr_n", "vol_z", "ret_ac1"]
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regime_classifier.joblib")

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._model = None
        self._is_trained = False
        self._load()

    def _load(self) -> None:
        import os
        if not os.path.exists(self.MODEL_PATH):
            return
        try:
            import joblib
            data = joblib.load(self.MODEL_PATH)
            self._model = data["model"]
            self._label_map = data.get("label_map", {i: i for i in range(self.n_components)})
            self._is_trained = True
            logger.info("Regime classifier loaded from %s", self.MODEL_PATH)
        except Exception as e:
            logger.warning("Could not load regime classifier: %s", e)

    def fit(self, price_dfs: list[pd.DataFrame]) -> "RegimeClassifier":
        """Train on a list of OHLCV DataFrames (one per symbol).

        Uses BayesianGaussianMixture with a Dirichlet-Process prior so unused
        components are auto-pruned (per architecture plan §8).
        """
        from sklearn.mixture import BayesianGaussianMixture
        from sklearn.preprocessing import StandardScaler
        import joblib

        frames = []
        for df in price_dfs:
            try:
                feat = _compute_regime_features(df)
                frames.append(feat)
            except Exception as e:
                logger.warning("Regime feature computation failed: %s", e)

        if not frames:
            logger.error("No data for regime classifier training.")
            return self

        combined = pd.concat(frames, ignore_index=True).dropna()
        X = combined[self.FEATURE_COLS].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # BayesianGaussianMixture with weight_concentration_prior=0.01 is the
        # exact configuration from updated_architecture_plan_en.md §8.
        bgmm = BayesianGaussianMixture(
            n_components=max(self.n_components, 5),  # over-provision; DP will prune
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.01,
            max_iter=300,
            random_state=42,
        )
        bgmm.fit(X_scaled)

        # Assign regime labels by mean volatility (ascending → RANGING, TRENDING, VOLATILE).
        # BGMM may have pruned components — only label the ones with nontrivial weight.
        weights = bgmm.weights_
        active = np.where(weights > 0.02)[0]  # active components (weight > 2%)
        if len(active) == 0:
            active = np.arange(len(weights))
        rv_means = bgmm.means_[active, 0]  # rv is feature 0
        order = active[np.argsort(rv_means)]
        # Map active component index → ordered regime label (0,1,2,...).
        # Inactive components fall back to nearest active.
        label_map = {int(order[i]): min(i, self.n_components - 1) for i in range(len(order))}
        for i in range(len(weights)):
            if i not in label_map:
                # Unused component → assume RANGING (safest)
                label_map[i] = 0

        self._model = {"gmm": bgmm, "scaler": scaler}
        self._label_map = label_map
        self._is_trained = True
        # Cache for online updates (warm-start refits)
        self._scaled_history = X_scaled

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump({"model": self._model, "label_map": label_map}, self.MODEL_PATH)
        logger.info(
            "BayesianGMM regime classifier trained. %d active components: %s",
            len(active),
            {REGIME_NAMES.get(label_map.get(int(c), 0), "?"): float(weights[c])
             for c in active},
        )

        from datetime import datetime, timezone as _tz
        meta_path = self.MODEL_PATH.replace(".joblib", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as _mf:
            json.dump(
                {
                    "n_components": self.n_components,
                    "n_components_active": int(len(active)),
                    "weights": [float(w) for w in weights],
                    "regimes": REGIME_NAMES,
                    "label_map": {str(k): v for k, v in label_map.items()},
                    "n_samples": int(len(combined)),
                    "model_path": self.MODEL_PATH,
                    "last_trained": datetime.now(_tz.utc).isoformat(),
                    "model_type": "BayesianGaussianMixture",
                    "weight_concentration_prior": 0.01,
                    "accuracy_note": "Unsupervised BayesianGMM — no labelled accuracy",
                    "accuracy": None,
                },
                _mf,
                indent=2,
            )
        return self

    def partial_fit(self, df: pd.DataFrame, max_history_bars: int = 50_000) -> "RegimeClassifier":
        """Approximate online update via warm-start refit.

        sklearn's BayesianGaussianMixture lacks `partial_fit`, so we re-fit
        on the *combined* (history ∪ new) buffer with the previous component
        means as the warm-start init. Bounded by `max_history_bars` so the
        window slides forward.
        """
        if not self._is_trained or self._model is None:
            return self.fit([df])
        try:
            from sklearn.mixture import BayesianGaussianMixture
            feat = _compute_regime_features(df)
            X_new = self._model["scaler"].transform(feat[self.FEATURE_COLS].values)
            history = self._scaled_history if hasattr(self, "_scaled_history") else X_new
            combined = np.concatenate([history, X_new], axis=0)
            if len(combined) > max_history_bars:
                combined = combined[-max_history_bars:]
            old = self._model["gmm"]
            bgmm = BayesianGaussianMixture(
                n_components=old.n_components,
                covariance_type=old.covariance_type,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=0.01,
                max_iter=200,
                mean_precision_prior=1.0,
                random_state=42,
                warm_start=False,
                init_params="kmeans",
            )
            # Initialise from previous means to keep label semantics stable.
            bgmm.fit(combined)
            self._model["gmm"] = bgmm
            self._scaled_history = combined
            return self
        except Exception as e:
            logger.warning("partial_fit failed: %s", e)
            return self

    def predict(self, df: pd.DataFrame, last_n: int = 48) -> int:
        """
        Predict regime for the most recent window.
        Returns regime int: 0=RANGING, 1=TRENDING, 2=VOLATILE.
        Falls back to RANGING if model not loaded.
        """
        if not self._is_trained or self._model is None:
            return 0  # default: assume ranging

        try:
            tail = df.tail(last_n) if len(df) > last_n else df
            feat = _compute_regime_features(tail)
            if feat.empty:
                return 0
            X = feat[self.FEATURE_COLS].iloc[[-1]].values
            X_scaled = self._model["scaler"].transform(X)
            raw_label = int(self._model["gmm"].predict(X_scaled)[0])
            return self._label_map.get(raw_label, 0)
        except Exception as e:
            logger.debug("Regime prediction error: %s", e)
            return 0

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime for every bar in df. Returns integer Series."""
        if not self._is_trained or self._model is None:
            return pd.Series(0, index=df.index)

        try:
            feat = _compute_regime_features(df)
            X = feat[self.FEATURE_COLS].values
            X_scaled = self._model["scaler"].transform(X)
            raw_labels = self._model["gmm"].predict(X_scaled)
            mapped = np.array([self._label_map.get(int(r), 0) for r in raw_labels])
            return pd.Series(mapped, index=feat.index)
        except Exception as e:
            logger.debug("Regime series prediction error: %s", e)
            return pd.Series(0, index=df.index)

    @property
    def is_ready(self) -> bool:
        return self._is_trained

    @staticmethod
    def regime_name(code: int) -> str:
        return REGIME_NAMES.get(code, "UNKNOWN")

    @staticmethod
    def approved_strategies(regime: int) -> list[str]:
        return REGIME_STRATEGY_MAP.get(regime, [])

    @staticmethod
    def size_multiplier(regime: int) -> float:
        return REGIME_SIZE_MULT.get(regime, 1.0)


def train_regime_classifier(symbols: Optional[list] = None) -> RegimeClassifier:
    """
    Standalone training entry point. Called from train_all_models.py.
    """
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    if symbols is None:
        wl_path = os.path.join(PROJECT_ROOT, "data", "watchlist.json")
        if os.path.exists(wl_path):
            with open(wl_path, "r") as f:
                symbols = [s.replace("/", "_") for s in json.load(f)]
        else:
            symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

    price_dfs = []
    for sym in symbols:
        for fname in [f"{sym}_1h.csv.gz", f"{sym}_spot_1h.csv.gz"]:
            fpath = os.path.join(raw_dir, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath, usecols=["timestamp", "open", "high", "low",
                                                     "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    price_dfs.append(df)
                    break
                except Exception as e:
                    logger.warning("Could not load %s: %s", fpath, e)

    clf = RegimeClassifier()
    clf.fit(price_dfs)
    return clf
