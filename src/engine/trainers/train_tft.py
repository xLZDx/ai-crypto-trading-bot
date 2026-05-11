"""src.engine.trainers.train_tft — wrapper for TFT Neural (GPU)."""
from __future__ import annotations

from ._common import run_trainer
from .types import TrainingResult


def train(timeframe: str, *, force: bool = False,
          symbol: str = "BTC/USDT", **kwargs) -> TrainingResult:
    from src.engine.train_tft_model import train_tft_model
    return run_trainer('tft', timeframe, symbol=symbol,
                        train_fn=train_tft_model, **kwargs)
