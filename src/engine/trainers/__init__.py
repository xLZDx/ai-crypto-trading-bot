"""src.engine.trainers — Sprint 1a R1 trainers package.

Per-model thin wrappers around existing trainer modules (src.engine.train_model,
train_trend_model, train_futures_model, train_scalping_model, train_tft_model,
train_meta_labeler, src.analysis.regime_classifier, src.training.joint_oft_rl).

Each wrapper exposes a uniform contract:

    def train(timeframe: str, *, force: bool = False, symbol: str = "BTC/USDT",
              **kwargs) -> TrainingResult

Returning a TrainingResult dataclass (see .types). This standardizes the
output across every model — required by Sprint 1a R2 (KPI gate) to enforce
the same threshold logic regardless of which trainer ran.

The existing modules (src/engine/train_*_model.py) keep their original
signatures; wrappers only ADAPT them. Operators using the legacy CLI
(python -m src.engine.train_all_models) get the same behavior; only the
KPI gate / cluster routing path goes through the wrappers.

Registry: `TRAINER_REGISTRY` maps dashboard model_key → train_fn. Single
source of truth — dashboard, pipeline_orchestrator, worker.py, master_agent
all import from here.
"""
from __future__ import annotations

from .types import TrainingResult
from . import train_base
from . import train_trend
from . import train_futures
from . import train_scalping
from . import train_meta
from . import train_tft
from . import train_oft
from . import train_regime


# Single source of truth for "what is the training function for model_key X".
# Previously this map was duplicated across:
#   - src/dashboard/app.py:_TRAINER_DISPATCH
#   - src/engine/train_all_models.py:_train_all_inner (inline calls)
#   - src/training/distributed/worker.py:handlers
# All three should migrate to importing TRAINER_REGISTRY here.
TRAINER_REGISTRY = {
    'base':     train_base.train,
    'trend':    train_trend.train,
    'futures':  train_futures.train,
    'scalping': train_scalping.train,
    'meta':     train_meta.train,
    'tft':      train_tft.train,
    'oft':      train_oft.train,
    'regime':   train_regime.train,
}

__all__ = ['TrainingResult', 'TRAINER_REGISTRY']
