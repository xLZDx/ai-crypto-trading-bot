"""
Behavioral tests for src/agents/trainers/base_trainer_agent.py — Sprint 1A R1 template.

Covers:
  - MODEL_KEY must be set (subclass with empty MODEL_KEY raises)
  - Concrete subclass that succeeds → returns ok=True, meta path, started/finished
  - Subclass that raises → ok=False, error captured, no crash
  - train_async returns a thread
  - last_result() returns the most recent train_now outcome
  - KPI gate post-flight is invoked on success
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_missing_model_key_raises():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class BadAgent(BaseTrainerAgent):
        pass  # MODEL_KEY = '' (inherited)

    with pytest.raises(ValueError, match='MODEL_KEY'):
        BadAgent()


def test_successful_train_returns_ok():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class FakeAgent(BaseTrainerAgent):
        MODEL_KEY = 'fake'
        def _do_train(self, timeframe: str, **kwargs):
            return '/fake/meta.json'

    a = FakeAgent()
    with patch('src.engine.kpi_gate.evaluate_from_meta_json',
               return_value={'persisted': True, 'retired_now': False}):
        r = a.train_now('1h')
    assert r['ok'] is True
    assert r['meta_json_path'] == '/fake/meta.json'
    assert r['started_at']
    assert r['finished_at']
    assert r['error'] is None
    assert r['kpi_gate']['persisted'] is True


def test_failed_train_captures_error():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class FailingAgent(BaseTrainerAgent):
        MODEL_KEY = 'failing'
        def _do_train(self, timeframe: str, **kwargs):
            raise RuntimeError('simulated training failure')

    a = FailingAgent()
    r = a.train_now('1h')
    assert r['ok'] is False
    assert 'simulated training failure' in r['error']


def test_train_returning_none_marks_not_ok():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class NoneAgent(BaseTrainerAgent):
        MODEL_KEY = 'none'
        def _do_train(self, timeframe: str, **kwargs):
            return None

    a = NoneAgent()
    r = a.train_now('1h')
    assert r['ok'] is False
    assert r['meta_json_path'] is None


def test_train_async_returns_thread():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class FastAgent(BaseTrainerAgent):
        MODEL_KEY = 'fast'
        def _do_train(self, timeframe: str, **kwargs):
            return '/fake/meta.json'

    a = FastAgent()
    with patch('src.engine.kpi_gate.evaluate_from_meta_json',
               return_value={'persisted': True, 'retired_now': False}):
        thread = a.train_async('1h')
        thread.join(timeout=5.0)
    last = a.last_result()
    assert last is not None
    assert last['ok'] is True


def test_last_result_initially_none():
    from src.agents.trainers.base_trainer_agent import BaseTrainerAgent

    class IdleAgent(BaseTrainerAgent):
        MODEL_KEY = 'idle'
        def _do_train(self, timeframe: str, **kwargs):
            return '/fake/meta.json'

    a = IdleAgent()
    assert a.last_result() is None


def test_trainer_meta_agent_imports():
    """Sanity check that the reference TrainerMetaAgent class imports OK."""
    from src.agents.trainers.base_trainer_agent import TrainerMetaAgent
    assert TrainerMetaAgent.MODEL_KEY == 'meta'
    # Don't actually call _do_train — it would invoke a real training run.


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
