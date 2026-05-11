"""src.engine.trainers.train_oft — wrapper for OFT (Microstructure, exclusive lane).

OFT has a different signature than the rest: train_oft(symbol, timeframe, ...)
not train_fn(timeframe=...). This wrapper adapts it so callers see a uniform
TrainingResult-returning interface like every other trainer.
"""
from __future__ import annotations

import time
import traceback

from ._common import _read_meta, _populate_from_meta
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    """OFT trains on L2/L3 order-book microstructure. Canonical TF is 1m;
    higher TFs lose the microstructure detail. Multi-symbol is the norm —
    but this single-call wrapper handles ONE (symbol, tf) for compatibility
    with the cluster's per-task dispatch model. Operator wanting the full
    canonical 3-symbol sweep calls this 3 times via the cluster.
    """
    result = TrainingResult(
        model_key='oft', tf=timeframe, symbol=symbol,
        started_at=time.time(),
    )
    try:
        from src.training.joint_oft_rl import train_oft as _train_oft
        _train_oft(symbol, timeframe, **kwargs)
    except Exception as exc:
        result.error = f'{type(exc).__name__}: {exc}'
        result.extras['traceback_tail'] = traceback.format_exc()[-1024:]
    finally:
        result.finished_at = time.time()
        result.elapsed_s   = round(result.finished_at - result.started_at, 2)
    if result.ok:
        _populate_from_meta(result, _read_meta('oft', timeframe))
    return result
