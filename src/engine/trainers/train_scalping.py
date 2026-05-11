"""src.engine.trainers.train_scalping — wrapper for Scalping RF (1m)."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_scalping_model import train_scalping_model
    return run_trainer('scalping', timeframe, symbol=symbol,
                        train_fn=train_scalping_model, **kwargs)
