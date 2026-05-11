"""src.engine.trainers.train_base — wrapper for Base RF (1h SPOT)."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_model import train_model
    return run_trainer('base', timeframe, symbol=symbol,
                        train_fn=train_model, **kwargs)
