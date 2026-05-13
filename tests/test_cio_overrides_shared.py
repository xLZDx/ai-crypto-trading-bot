"""
Behavioral tests for src/utils/cio_overrides.py (shared helper).
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
def isolated_rules(tmp_path, monkeypatch):
    from src.utils import cio_overrides as co
    monkeypatch.setattr(co, 'TRAINING_RULES_PATH', tmp_path / 'training_rules.json')
    return co, tmp_path


def test_missing_file_returns_empty(isolated_rules):
    co, _ = isolated_rules
    assert co.load_cio_overrides('meta') == {}


def test_strips_audit_metadata(isolated_rules):
    co, tmp = isolated_rules
    (tmp / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta': {
                'cio_overrides': {
                    'pt_multiplier': 2.5,
                    '_applied_at': '2026-05-13T00:00:00+00:00',
                    '_study': 'foo',
                    '_best_value': 1.0,
                }
            }
        }
    }))
    result = co.load_cio_overrides('meta')
    assert result == {'pt_multiplier': 2.5}
    assert all(not k.startswith('_') for k in result)


def test_unknown_model_returns_empty(isolated_rules):
    co, tmp = isolated_rules
    (tmp / 'training_rules.json').write_text(json.dumps({'models': {}}))
    assert co.load_cio_overrides('does_not_exist') == {}


def test_malformed_json_returns_empty(isolated_rules):
    co, tmp = isolated_rules
    (tmp / 'training_rules.json').write_text('{not json')
    assert co.load_cio_overrides('meta') == {}


def test_per_model_isolation(isolated_rules):
    co, tmp = isolated_rules
    (tmp / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta':  {'cio_overrides': {'pt_multiplier': 2.5}},
            'base':  {'cio_overrides': {'n_estimators': 200}},
            'trend': {'cio_overrides': {'max_depth': 6}},
        }
    }))
    assert co.load_cio_overrides('meta')['pt_multiplier'] == 2.5
    assert co.load_cio_overrides('base')['n_estimators'] == 200
    assert co.load_cio_overrides('trend')['max_depth'] == 6


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
