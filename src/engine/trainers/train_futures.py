"""src.engine.trainers.train_futures — wrapper for Futures Short RF."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_futures_model import train_futures_model
    return run_trainer('futures', timeframe, symbol=symbol,
                        train_fn=train_futures_model, **kwargs)
