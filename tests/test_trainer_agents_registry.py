"""
Tests for the trainer-agent registry + factory (Sprint 1A R1 stubs).

Covers:
  - All 5 concrete agents present in TRAINER_AGENT_REGISTRY
  - Each agent has the right MODEL_KEY
  - get_trainer_agent(known) returns a fresh instance
  - get_trainer_agent(unknown) raises KeyError with helpful message
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_registry_has_5_agents():
    from src.agents.trainers.base_trainer_agent import TRAINER_AGENT_REGISTRY
    assert set(TRAINER_AGENT_REGISTRY.keys()) == {'meta', 'base', 'trend', 'futures', 'scalping'}


def test_each_agent_has_correct_model_key():
    from src.agents.trainers.base_trainer_agent import (
        TrainerMetaAgent, TrainerBaseAgent, TrainerTrendAgent,
        TrainerFuturesAgent, TrainerScalpingAgent,
    )
    assert TrainerMetaAgent.MODEL_KEY     == 'meta'
    assert TrainerBaseAgent.MODEL_KEY     == 'base'
    assert TrainerTrendAgent.MODEL_KEY    == 'trend'
    assert TrainerFuturesAgent.MODEL_KEY  == 'futures'
    assert TrainerScalpingAgent.MODEL_KEY == 'scalping'


def test_factory_returns_concrete_instance():
    from src.agents.trainers.base_trainer_agent import (
        get_trainer_agent, BaseTrainerAgent, TrainerMetaAgent,
    )
    a = get_trainer_agent('meta')
    assert isinstance(a, TrainerMetaAgent)
    assert isinstance(a, BaseTrainerAgent)


def test_factory_returns_independent_instances():
    """Factory yields fresh instances, not a shared singleton."""
    from src.agents.trainers.base_trainer_agent import get_trainer_agent
    a = get_trainer_agent('base')
    b = get_trainer_agent('base')
    assert a is not b


def test_factory_unknown_model_raises_keyerror():
    from src.agents.trainers.base_trainer_agent import get_trainer_agent
    with pytest.raises(KeyError, match='No trainer agent'):
        get_trainer_agent('not_a_real_model')


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
