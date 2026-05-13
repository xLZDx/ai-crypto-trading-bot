"""
Sprint 1A R1 — Base class for per-model trainer agents.

REFERENCE TEMPLATE. The full Sprint 1A R1 refactor (one file per model,
one supervised agent per model, topic-based dispatch) is a multi-day
refactor that needs its own dedicated session. This module ships the
contract + the BaseTrainerAgent skeleton so the upcoming per-model
agents have a clear parent class to inherit from.

What's done here:
  - BaseTrainerAgent class with lifecycle hooks (start / stop / heartbeat)
  - Standard contract for `train()` invocation result (TrainingResult)
  - Integration with KPI gate's evaluate_run after every successful train
  - Topic subscription stubs (real bus wiring is per-agent)

What's NOT done (deferred to R1 full sprint):
  - Per-model concrete agent files (TrainerBaseAgent, TrainerTrendAgent, etc.)
  - Removal of train_all_models.py monolith
  - master_agent integration for per-agent supervision
  - Topic-bus message format (Sprint 1A R1.4)

Use:
  from src.agents.trainers.base_trainer_agent import BaseTrainerAgent
  class TrainerMetaAgent(BaseTrainerAgent):
      MODEL_KEY = 'meta'
      def _do_train(self, timeframe: str, **kwargs):
          from src.engine.train_meta_labeler import train_meta_labeler
          return train_meta_labeler(timeframe=timeframe, **kwargs)
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseTrainerAgent:
    """
    Lifecycle:
      __init__(model_key)  → instance registered
      start()              → spawns the supervision thread (no auto-train)
      stop()               → graceful shutdown
      train_now(tf, **kw)  → blocking single-shot train; persists KPI to gate
      train_async(tf, **kw)→ fire-and-forget thread that calls train_now

    Subclasses must:
      1. Set class attribute MODEL_KEY (one of base/trend/futures/scalping/
         meta/regime/tft/oft).
      2. Implement _do_train(timeframe, **kwargs) — calls the concrete
         training function and returns the trained model's meta JSON path.
    """

    MODEL_KEY: str = ''  # subclass override

    def __init__(self):
        if not self.MODEL_KEY:
            raise ValueError(f'{self.__class__.__name__}.MODEL_KEY must be set')
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_heartbeat: float = 0.0
        self._last_result: dict | None = None
        self._lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f'trainer-{self.MODEL_KEY}', daemon=True,
        )
        self._thread.start()
        logger.info('[%s] trainer agent started', self.MODEL_KEY)

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        """Heartbeat loop — does NOT auto-train. Concrete agents subscribe
        to a 'train' topic in _setup_subscriptions (override in subclass)."""
        self._setup_subscriptions()
        while self._running:
            self._last_heartbeat = time.time()
            time.sleep(30.0)

    def _setup_subscriptions(self) -> None:
        """Subclass hook — wire bus subscriptions for the model's topic.
        Default: no-op (manual train_now calls only)."""

    # ── Training entry points ────────────────────────────────────────────────

    def train_now(self, timeframe: str, **kwargs) -> dict:
        """
        Blocking single-shot train. Returns:
          {
            'ok': bool,
            'model_key': str,
            'timeframe': str,
            'started_at': ISO,
            'finished_at': ISO,
            'meta_json_path': str | None,
            'error': str | None,
            'kpi_gate': {...},   // from kpi_gate.evaluate_from_meta_json
          }
        """
        started = datetime.now(timezone.utc)
        result = {
            'ok': False,
            'model_key': self.MODEL_KEY,
            'timeframe': timeframe,
            'started_at': started.isoformat(),
            'finished_at': None,
            'meta_json_path': None,
            'error': None,
        }
        try:
            meta_path = self._do_train(timeframe, **kwargs)
            result['meta_json_path'] = str(meta_path) if meta_path else None
            result['ok'] = bool(meta_path)
        except Exception as e:
            logger.error('[%s] train_now failed: %s\n%s',
                         self.MODEL_KEY, e, traceback.format_exc())
            result['error'] = str(e)
        result['finished_at'] = datetime.now(timezone.utc).isoformat()

        # KPI gate integration — persists a TrainingResult and auto-retires
        if result['ok'] and result['meta_json_path']:
            try:
                from src.engine import kpi_gate as kg
                result['kpi_gate'] = kg.evaluate_from_meta_json(
                    model_key=self.MODEL_KEY,
                    tf=timeframe,
                    meta_json_path=result['meta_json_path'],
                )
            except Exception as e:
                logger.warning('[%s] KPI gate post-flight skipped: %s',
                               self.MODEL_KEY, e)

        with self._lock:
            self._last_result = result
        return result

    def train_async(self, timeframe: str, **kwargs) -> threading.Thread:
        t = threading.Thread(
            target=self.train_now, args=(timeframe,), kwargs=kwargs,
            name=f'trainer-{self.MODEL_KEY}-{timeframe}', daemon=True,
        )
        t.start()
        return t

    def last_result(self) -> dict | None:
        with self._lock:
            return self._last_result

    # ── Abstract: subclass implements ────────────────────────────────────────

    @abstractmethod
    def _do_train(self, timeframe: str, **kwargs) -> Path | str | None:
        """
        Concrete training step. Must return the path to the model's meta JSON
        on success, or None on failure.

        Subclasses typically delegate to the existing `train_*` function in
        src/engine/*.py. Example:

            from src.engine.train_meta_labeler import train_meta_labeler
            train_meta_labeler(timeframe=timeframe, **kwargs)
            from src.utils.model_paths import artifact_paths
            return artifact_paths(self.MODEL_KEY, timeframe).get('meta')
        """
        raise NotImplementedError


# ── Reference implementations — minimal wrappers around existing trainers ──

def _resolve_meta_path(model_key: str, timeframe: str) -> Path | str | None:
    """Common helper — derive the meta JSON path via model_paths.artifact_paths."""
    try:
        from src.utils.model_paths import artifact_paths
        paths = artifact_paths(model_key, timeframe)
        return paths.get('meta')
    except Exception:
        return None


class TrainerMetaAgent(BaseTrainerAgent):
    """Concrete agent for the meta-labeler. Supports CIO override kwargs."""

    MODEL_KEY = 'meta'

    def _do_train(self, timeframe: str, **kwargs) -> Path | str | None:
        from src.engine.train_meta_labeler import train_meta_labeler
        train_meta_labeler(timeframe=timeframe, **kwargs)
        return _resolve_meta_path(self.MODEL_KEY, timeframe)


class TrainerBaseAgent(BaseTrainerAgent):
    """Concrete agent for the base directional model. Reads HP from
    training_rules.json (incl. cio_overrides merge)."""

    MODEL_KEY = 'base'

    def _do_train(self, timeframe: str, **kwargs) -> Path | str | None:
        from src.engine.train_model import train_model
        # train_model has no kwargs beyond timeframe; cio_overrides flow
        # through _load_model_params automatically.
        train_model(timeframe=timeframe)
        return _resolve_meta_path(self.MODEL_KEY, timeframe)


class TrainerTrendAgent(BaseTrainerAgent):
    """Concrete agent for the trend-follower."""

    MODEL_KEY = 'trend'

    def _do_train(self, timeframe: str, **kwargs) -> Path | str | None:
        from src.engine.train_trend_model import train_trend_model
        train_trend_model(timeframe=timeframe)
        return _resolve_meta_path(self.MODEL_KEY, timeframe)


class TrainerFuturesAgent(BaseTrainerAgent):
    """Concrete agent for the futures short classifier."""

    MODEL_KEY = 'futures'

    def _do_train(self, timeframe: str, **kwargs) -> Path | str | None:
        from src.engine.train_futures_model import train_futures_model
        train_futures_model(timeframe=timeframe)
        return _resolve_meta_path(self.MODEL_KEY, timeframe)


class TrainerScalpingAgent(BaseTrainerAgent):
    """Concrete agent for the scalping (1m) model.

    NOTE: scalping is fixed at 1m (train_scalping_model docstring is
    explicit about this). The timeframe arg is accepted but ignored by
    the underlying trainer.
    """

    MODEL_KEY = 'scalping'

    def _do_train(self, timeframe: str = '1m', **kwargs) -> Path | str | None:
        from src.engine.train_scalping_model import train_scalping_model
        train_scalping_model(timeframe='1m')
        return _resolve_meta_path(self.MODEL_KEY, '1m')


# ── Registry — used by the orchestrator to dispatch by model_key ────────────

TRAINER_AGENT_REGISTRY: dict[str, type[BaseTrainerAgent]] = {
    'meta':     TrainerMetaAgent,
    'base':     TrainerBaseAgent,
    'trend':    TrainerTrendAgent,
    'futures':  TrainerFuturesAgent,
    'scalping': TrainerScalpingAgent,
}


def get_trainer_agent(model_key: str) -> BaseTrainerAgent:
    """Factory — returns a fresh agent instance for `model_key`.

    Raises KeyError if the model_key has no registered agent class yet
    (e.g., 'tft', 'oft', 'regime' — those are queued for future R1 sub-tasks
    since their trainers have different interfaces and resource constraints).
    """
    cls = TRAINER_AGENT_REGISTRY.get(model_key)
    if cls is None:
        raise KeyError(
            f"No trainer agent registered for model_key={model_key!r}. "
            f"Registered: {sorted(TRAINER_AGENT_REGISTRY.keys())}"
        )
    return cls()
