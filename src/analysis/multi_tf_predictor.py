"""
multi_tf_predictor — load every per-TF artifact for a model key and route
predictions to the correct timeframe.

Phase G of the institutional roadmap. PR 2 made the trainer multi-TF
(writing models/<key>_<tf>_*.{joblib,json}), but the bot's inference path
still loaded only the canonical filename (e.g. btc_rf_model.joblib for
'base' @ 1h). This wrapper closes the loop:

  - Loads the canonical (1h) model via the legacy filename — guarantees
    backwards-compat with every caller that does `.predict(data)` without
    naming a TF.
  - Auto-discovers and loads every additional per-TF artifact present on
    disk via list_per_tf_artifacts() — so dropping a `base_4h_model.joblib`
    into models/ extends inference with no code change.
  - `predict_at(tf, data)` runs inference at a specific TF.
  - `predict_all(data_by_tf)` runs inference at every loaded TF when the
    caller has multi-TF feature frames available.
  - Old single-TF `.predict(data)` keeps working — routes to canonical.

Used by main.py (replacing the bare MLPredictor instantiation) and by the
strategy_registry once PR 12 wires per-strategy TF auto-selection.

Public surface:
  MultiTFPredictor(key, model_type=None)
      .predict(data)             -> int|None      (canonical-TF backwards compat)
      .predict_at(tf, data)      -> int|None      (TF-specific)
      .predict_all(data_by_tf)   -> dict[tf, int]
      .available_tfs()           -> list[str]
      .is_loaded                 -> bool
      .accuracy / long_accuracy / short_accuracy   (forwarded from canonical)
      .last_status / last_error                     (forwarded from canonical)
      .by_tf[tf]                                   (raw MLPredictor for advanced use)
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.analysis.ml_predictor import MLPredictor
from src.utils.model_paths import (
    CANONICAL_TF, KEYS, LEGACY_MODEL_NAME,
    list_per_tf_artifacts,
)

logger = logging.getLogger(__name__)

# Only the four tabular ML model families have per-TF variants — TFT/OFT
# are single-TF by design; meta/regime aggregate over TFs upstream.
_MULTI_TF_KEYS = ("base", "trend", "futures", "scalping")


class MultiTFPredictor:
    """Wrapper around N MLPredictors keyed by timeframe."""

    def __init__(self, key: str, model_type: str | None = None):
        if key not in KEYS:
            raise ValueError(f"unknown model key {key!r}; valid: {sorted(KEYS)}")
        self.key = key
        self.model_type = model_type or key
        self._canonical_tf = CANONICAL_TF[key]
        self._predictors: dict[str, MLPredictor] = {}

        # 1. Always load canonical via the legacy filename. Even if a
        #    matching per-TF file exists, we use the legacy name so the
        #    canonical predictor stays the single source of truth for
        #    backwards-compat callers.
        legacy_name = LEGACY_MODEL_NAME[key]
        self._predictors[self._canonical_tf] = MLPredictor(
            model_filename=legacy_name, model_type=self.model_type
        )

        # 2. Auto-discover per-TF artifacts. Skip the canonical TF so we
        #    don't double-load it.
        for tf, model_path, _meta_path in list_per_tf_artifacts(key):
            if tf == self._canonical_tf:
                continue
            try:
                self._predictors[tf] = MLPredictor(
                    model_filename=model_path.name,
                    model_type=self.model_type,
                )
            except Exception as exc:
                logger.warning("multi_tf_predictor: failed to load %s @ %s: %s",
                               key, tf, exc)

        loaded = [tf for tf, p in self._predictors.items() if p.is_loaded]
        if loaded:
            logger.info("MultiTFPredictor[%s] loaded TFs=%s", key, loaded)
        else:
            logger.warning("MultiTFPredictor[%s] loaded NO TFs (canonical legacy missing too)", key)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, data) -> int | None:
        """Backwards-compat single-TF prediction. Routes to canonical TF."""
        return self._predictors[self._canonical_tf].predict(data)

    def predict_at(self, tf: str, data) -> int | None:
        """Run inference at a specific timeframe.

        Returns None if no predictor is loaded for that TF — callers can
        fall back to the canonical TF themselves or treat as no signal.
        """
        p = self._predictors.get(tf)
        if p is None or not p.is_loaded:
            return None
        return p.predict(data)

    def predict_all(self, data_by_tf: dict) -> dict[str, int | None]:
        """Run inference at every loaded TF. `data_by_tf` is {tf: features}.

        Returns {tf: prediction} for every TF that has BOTH a loaded
        predictor AND data in `data_by_tf`. TFs missing data are skipped.
        """
        out: dict[str, int | None] = {}
        for tf, p in self._predictors.items():
            if not p.is_loaded:
                continue
            d = data_by_tf.get(tf)
            if d is None:
                continue
            out[tf] = p.predict(d)
        return out

    def available_tfs(self) -> list[str]:
        """Sorted list of timeframes for which a model is loaded."""
        return sorted([tf for tf, p in self._predictors.items() if p.is_loaded])

    @property
    def by_tf(self) -> dict[str, MLPredictor]:
        """Raw access to the underlying MLPredictors. Read-only contract:
        callers SHOULD NOT mutate this dict."""
        return self._predictors

    # ── Backwards-compat passthroughs (canonical TF) ──────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._predictors[self._canonical_tf].is_loaded

    @property
    def accuracy(self) -> float:
        return self._predictors[self._canonical_tf].accuracy

    @property
    def long_accuracy(self) -> float:
        return self._predictors[self._canonical_tf].long_accuracy

    @property
    def short_accuracy(self) -> float:
        return self._predictors[self._canonical_tf].short_accuracy

    @property
    def last_status(self) -> str:
        return self._predictors[self._canonical_tf].last_status

    @property
    def last_error(self) -> str:
        return self._predictors[self._canonical_tf].last_error

    @property
    def model_path(self) -> str:
        return self._predictors[self._canonical_tf].model_path

    @property
    def model(self):
        return self._predictors[self._canonical_tf].model

    def _get_model_features(self):
        """Forward to canonical predictor — meta-labeler trainer uses this."""
        return self._predictors[self._canonical_tf]._get_model_features()


__all__ = ["MultiTFPredictor"]
