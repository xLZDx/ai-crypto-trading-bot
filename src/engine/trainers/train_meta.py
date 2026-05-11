"""src.engine.trainers.train_meta — wrapper for Meta-Labeler."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_meta_labeler import train_meta_labeler
    return run_trainer('meta', timeframe, symbol=symbol,
                        train_fn=train_meta_labeler, **kwargs)
