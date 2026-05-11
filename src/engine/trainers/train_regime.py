"""src.engine.trainers.train_regime — wrapper for Regime Classifier (GMM).

Regime uses TF-invariant features (vol, ADX, returns z-score); one canonical
TF (1h) is enough. train_regime_classifier() takes NO timeframe argument —
this wrapper accepts the tf for API uniformity but ignores it when calling
the underlying function (always trains the canonical 1h variant).
"""
from __future__ import annotations

import time
import traceback

from ._common import _read_meta, _populate_from_meta
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    result = TrainingResult(
        model_key='regime', tf=timeframe, symbol=symbol,
        started_at=time.time(),
    )
    try:
        from src.analysis.regime_classifier import train_regime_classifier
        clf = train_regime_classifier()
        if clf is not None and not getattr(clf, 'is_ready', True):
            result.error = 'regime classifier training produced no output (clf.is_ready=False)'
    except Exception as exc:
        result.error = f'{type(exc).__name__}: {exc}'
        result.extras['traceback_tail'] = traceback.format_exc()[-1024:]
    finally:
        result.finished_at = time.time()
        result.elapsed_s   = round(result.finished_at - result.started_at, 2)
    if result.ok:
        _populate_from_meta(result, _read_meta('regime', timeframe))
    return result
