"""
Tests for CIOAgent.apply_best — promotes the winning Optuna proposal into
data/training_rules.json under models[<target>].cio_overrides.

Verifies:
  - operator_approved=False is hard rejected
  - no proposals → error
  - unknown study_name → error
  - unknown target_model → error
  - happy path: best_params land under cio_overrides, backup file created
  - selects the HIGHEST best_value when multiple proposals match the study
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def cio_isolated(tmp_path, monkeypatch):
    """Repoint CIO paths to tmp; seed minimal rules + proposals."""
    from src.engine import cio_agent as ca
    rules_path = tmp_path / 'training_rules.json'
    proposals_path = tmp_path / 'cio_proposals.json'
    rules_path.write_text(json.dumps({
        'models': {
            'meta':  {'applicable_tfs': ['1h'], 'params': {'n_estimators': 100}},
            'base':  {'applicable_tfs': ['1h'], 'params': {'n_estimators': 200}},
        }
    }, indent=2))
    monkeypatch.setattr(ca, 'TRAINING_RULES_PATH', rules_path)
    monkeypatch.setattr(ca, 'CIO_PROPOSALS_PATH', proposals_path)
    return ca, rules_path, proposals_path


def _proposal(study='macro_parameter_search_v1', best_value=2.5, params=None):
    return {
        'study_name': study,
        'n_trials': 20,
        'best_value': best_value,
        'best_params': params or {'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                                  'timeframe': '1h'},
        'best_user_attrs': {},
        'completed_at': '2026-05-13T12:00:00+00:00',
    }


def test_apply_best_requires_operator_approval(cio_isolated):
    ca, _, proposals = cio_isolated
    proposals.write_text(json.dumps({'proposals': [_proposal()]}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    result = agent.apply_best(operator_approved=False)
    assert result['ok'] is False
    assert 'operator_approved' in result['error']


def test_apply_best_no_proposals_returns_error(cio_isolated):
    ca, _, proposals = cio_isolated
    proposals.write_text(json.dumps({'proposals': []}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    result = agent.apply_best(operator_approved=True)
    assert result['ok'] is False
    assert 'no proposals' in result['error']


def test_apply_best_unknown_study(cio_isolated):
    ca, _, proposals = cio_isolated
    proposals.write_text(json.dumps({'proposals': [_proposal(study='other_study')]}))
    agent = ca.CIOAgent(study_name='not_this_one')
    result = agent.apply_best(operator_approved=True)
    assert result['ok'] is False
    assert 'no proposal found' in result['error']


def test_apply_best_unknown_target_model(cio_isolated):
    ca, _, proposals = cio_isolated
    proposals.write_text(json.dumps({'proposals': [_proposal()]}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    result = agent.apply_best(operator_approved=True, target_model='does_not_exist')
    assert result['ok'] is False
    assert 'target_model' in result['error']


def test_apply_best_happy_path_writes_cio_overrides(cio_isolated):
    ca, rules_path, proposals_path = cio_isolated
    proposals_path.write_text(json.dumps({'proposals': [
        _proposal(best_value=1.5, params={'pt_multiplier': 2.0, 'sl_multiplier': 1.5}),
        _proposal(best_value=3.1, params={'pt_multiplier': 2.5, 'sl_multiplier': 1.5}),
    ]}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    result = agent.apply_best(operator_approved=True, target_model='meta')
    assert result['ok'] is True
    assert result['target_model'] == 'meta'
    # Highest best_value wins
    assert result['after']['pt_multiplier'] == 2.5
    assert result['after']['_best_value'] == 3.1

    # Rules file mutated
    rules = json.loads(rules_path.read_text())
    overrides = rules['models']['meta']['cio_overrides']
    assert overrides['pt_multiplier'] == 2.5
    assert overrides['sl_multiplier'] == 1.5
    assert '_applied_at' in overrides
    assert '_study' in overrides


def test_apply_best_creates_backup_file(cio_isolated):
    ca, rules_path, proposals_path = cio_isolated
    proposals_path.write_text(json.dumps({'proposals': [_proposal()]}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    result = agent.apply_best(operator_approved=True, target_model='meta')
    assert result['ok'] is True
    backup_name = result['backup']
    assert backup_name.startswith('training_rules.json.bak-')
    # Backup file exists in same directory
    backup_path = rules_path.parent / backup_name
    assert backup_path.exists()
    # And contains the pre-change state (no cio_overrides yet)
    pre = json.loads(backup_path.read_text())
    assert 'cio_overrides' not in pre['models']['meta']


def test_apply_best_idempotent_overwrites_overrides(cio_isolated):
    """Second apply with a different proposal replaces (not stacks) cio_overrides."""
    ca, rules_path, proposals_path = cio_isolated
    proposals_path.write_text(json.dumps({'proposals': [
        _proposal(best_value=1.0, params={'pt_multiplier': 2.0}),
    ]}))
    agent = ca.CIOAgent(study_name='macro_parameter_search_v1')
    r1 = agent.apply_best(operator_approved=True, target_model='meta')
    assert r1['after']['pt_multiplier'] == 2.0

    # Add a higher-value proposal and re-apply
    proposals = json.loads(proposals_path.read_text())
    proposals['proposals'].append(_proposal(best_value=5.0, params={'pt_multiplier': 3.0}))
    proposals_path.write_text(json.dumps(proposals))
    r2 = agent.apply_best(operator_approved=True, target_model='meta')
    assert r2['after']['pt_multiplier'] == 3.0
    assert r2['before']['pt_multiplier'] == 2.0  # before reflects r1's write


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
