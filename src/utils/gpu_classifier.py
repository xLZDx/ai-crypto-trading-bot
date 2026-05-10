"""GPU-aware tabular classifier wrapper.

Returns either an XGBoost-on-CUDA classifier (when GPU is available) or
a sklearn HistGradientBoostingClassifier fallback (when only CPU is
available — e.g. on a CPU-lane worker with CUDA_VISIBLE_DEVICES=''
set by the sweep coordinator).

Why a wrapper instead of editing every trainer to use XGBoost directly:

  - The dual-lane architecture (2026-05-10 commit 3eb6cd4) runs the
    SAME trainer code on both cpu-lane and gpu-lane workers. We need
    one source of truth that picks the right backend at runtime.
  - Hyperparameter names differ between sklearn and XGBoost. The
    wrapper accepts a single normalised hyperparameter dict and
    translates to whichever backend is selected.
  - A future migration to LightGBM / CatBoost can swap the backend
    without touching trainer files — the wrapper interface stays.

Hyperparameter mapping (normalised → sklearn HistGBT → XGBoost):

  n_estimators         → max_iter            n_estimators
  max_depth            → max_depth           max_depth
  learning_rate        → learning_rate       learning_rate
  l2_regularization    → l2_regularization   reg_lambda
  early_stopping       → early_stopping      early_stopping_rounds (when set)
  random_state         → random_state        random_state

Usage:
  from src.utils.gpu_classifier import make_classifier
  clf = make_classifier(n_estimators=300, max_depth=4, learning_rate=0.05)
  clf.fit(X, y, sample_weight=w)
  proba = clf.predict_proba(X_val)[:, 1]

Both backends expose .fit() and .predict_proba() with the same call
signature. .predict() also works.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Detect CUDA once per process (cheap; cached by torch).
def _cuda_available() -> bool:
    """True if torch+CUDA is importable and at least one GPU is visible.
    CUDA_VISIBLE_DEVICES='' on a CPU-lane worker forces this to False."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        # Empty string means "no GPUs visible" — explicit CPU-lane signal.
        # Note: an UNSET env var also returns "" from .get(default="");
        # we differentiate by checking the raw value below.
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _use_gpu_backend() -> bool:
    """Pick GPU iff BOTH torch+CUDA and xgboost are installed."""
    return _cuda_available() and _xgboost_available()


def make_classifier(
    n_estimators:      int   = 300,
    max_depth:         int   = 4,
    learning_rate:     float = 0.05,
    l2_regularization: float = 0.1,
    early_stopping:    bool  = False,
    random_state:      int   = 42,
    class_weight:      str | None = "balanced",
    **extra: Any,
):
    """Return a GPU-backed XGBoost classifier when CUDA is available,
    else fall back to sklearn HistGradientBoostingClassifier.

    Both returns expose .fit(X, y, sample_weight=...) and
    .predict_proba(X) with the same signature, so trainer code stays
    backend-agnostic.

    `class_weight='balanced'` is honoured natively by sklearn HistGBT
    and emulated for XGBoost by computing per-row sample weights at
    fit time (handled inside the wrapper class below)."""
    if _use_gpu_backend():
        return _XGBClassifierWrapper(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=l2_regularization,
            early_stopping=early_stopping,
            random_state=random_state,
            class_weight=class_weight,
            **extra,
        )
    # CPU fallback: sklearn HistGBT, the model that's been shipping today.
    from sklearn.ensemble import HistGradientBoostingClassifier
    kwargs = dict(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=random_state,
        early_stopping=early_stopping,
        class_weight=class_weight,
    )
    # Drop kwargs sklearn doesn't know.
    return HistGradientBoostingClassifier(**kwargs)


class _XGBClassifierWrapper:
    """Thin adapter around xgboost.XGBClassifier that:
      - uses tree_method='hist' + device='cuda' for GPU training
      - emulates class_weight='balanced' by computing per-row sample
        weights when the caller doesn't pass sample_weight explicitly
      - exposes .fit() and .predict_proba() with the same signature as
        sklearn HistGBT so callers don't care which backend is used

    Not subclassing sklearn's BaseEstimator on purpose — we don't need
    cross-validation/ pipeline integration, and this keeps the surface
    minimal."""

    def __init__(self, *, n_estimators: int, max_depth: int,
                 learning_rate: float, reg_lambda: float,
                 early_stopping: bool, random_state: int,
                 class_weight: str | None, **extra: Any):
        import xgboost as xgb
        self._class_weight = class_weight
        self._early_stopping = early_stopping
        self._classes_: Any = None
        params = {
            "n_estimators":  int(n_estimators),
            "max_depth":     int(max_depth),
            "learning_rate": float(learning_rate),
            "reg_lambda":    float(reg_lambda),
            "random_state":  int(random_state),
            "tree_method":   "hist",
            "device":        "cuda",
            # Binary classification — most of our tabular models are 2-class.
            # XGBoost autodetects but being explicit avoids a surprise.
            "objective":     "binary:logistic",
            "eval_metric":   "logloss",
        }
        if early_stopping:
            params["early_stopping_rounds"] = 20
        # Filter extras to known XGBoost params to avoid runtime errors.
        _xgb_known = {"min_child_weight", "subsample", "colsample_bytree",
                       "gamma", "scale_pos_weight"}
        for k, v in (extra or {}).items():
            if k in _xgb_known:
                params[k] = v
        self._clf = xgb.XGBClassifier(**params)

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Fit. If class_weight='balanced' and caller didn't pass
        sample_weight, compute per-row weights to balance classes —
        XGBoost has no native class_weight kwarg."""
        if sample_weight is None and self._class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y)
        kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            kwargs["eval_set"] = eval_set
        self._clf.fit(X, y, **kwargs)
        # sklearn-style classes_ for compatibility with downstream code.
        self._classes_ = self._clf.classes_
        return self

    @property
    def classes_(self):
        return self._classes_

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def n_iter_(self):
        # Map XGBoost's best_iteration / n_estimators to sklearn's n_iter_.
        try:
            return self._clf.best_iteration + 1
        except Exception:
            return self._clf.get_params().get("n_estimators", 0)

    def __repr__(self):
        return f"XGBClassifierWrapper(gpu=True, n_iter={self.n_iter_})"
