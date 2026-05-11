"""src.engine.trainers.train_trend — wrapper for Trend RF (SPOT)."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_trend_model import train_trend_model
    return run_trainer('trend', timeframe, symbol=symbol,
                        train_fn=train_trend_model, **kwargs)
