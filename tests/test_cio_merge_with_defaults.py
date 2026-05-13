"""
Tests for src/utils/cio_overrides.py:merge_with_defaults — X1.3 helper.

The trend/futures/scalping trainers now call this helper to merge operator-
approved CIO overrides on top of their hardcoded defaults. The contract:

  - No overrides for the model → return defaults unchanged + empty applied.
  - Valid overrides → apply them, return them in `applied`.
  - Wrong-type override → drop with WARN, default preserved.
  - Out-of-range override → drop with WARN, default preserved.
  - Audit metadata keys (`_applied_at` etc.) are filtered upstream by
    load_cio_overrides, so they never reach the schema check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = {
    'n_estimators':  (int,   1,    10_000),
    'max_depth':     (int,   1,    50),
    'learning_rate': (float, 1e-4, 1.0),
    'class_weight':  (str,   None, None),
}
DEFAULTS = {
    'n_estimators': 400, 'max_depth': 5,
    'learning_rate': 0.05, 'class_weight': 'balanced',
}


@pytest.fixture
def isolated_rules(tmp_path, monkeypatch):
    """Redirect TRAINING_RULES_PATH to a tmp file so each test starts clean."""
    from src.utils import cio_overrides as co
    rules_path = tmp_path / 'training_rules.json'
    monkeypatch.setattr(co, 'TRAINING_RULES_PATH', rules_path)
    return co, rules_path


def _write_rules(rules_path, model_key, cio_overrides):
    rules_path.write_text(json.dumps({
        '_version': 'test', 'models': {model_key: {'cio_overrides': cio_overrides}},
    }), encoding='utf-8')


def test_no_overrides_returns_defaults(isolated_rules):
    co, rules_path = isolated_rules
    # Rules file doesn't exist — load_cio_overrides returns {}
    merged, applied = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    assert merged == DEFAULTS
    assert applied == {}


def test_valid_overrides_merged(isolated_rules):
    co, rules_path = isolated_rules
    _write_rules(rules_path, 'trend', {
        'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.01,
    })
    merged, applied = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    assert merged['n_estimators']  == 800
    assert merged['max_depth']     == 10
    assert merged['learning_rate'] == 0.01
    assert merged['class_weight']  == 'balanced'   # default preserved
    assert applied == {'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.01}


def test_wrong_type_override_skipped(isolated_rules):
    co, rules_path = isolated_rules
    _write_rules(rules_path, 'trend', {'n_estimators': 'many', 'max_depth': 12})
    merged, applied = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    assert merged['n_estimators'] == 400  # default preserved
    assert merged['max_depth']    == 12   # valid override applied
    assert 'n_estimators' not in applied
    assert 'max_depth' in applied


def test_out_of_range_override_skipped(isolated_rules):
    co, rules_path = isolated_rules
    _write_rules(rules_path, 'trend', {'max_depth': 999, 'learning_rate': 0.1})
    merged, applied = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    assert merged['max_depth']     == 5    # default preserved (999 > 50 cap)
    assert merged['learning_rate'] == 0.1  # valid override applied
    assert 'max_depth' not in applied
    assert 'learning_rate' in applied


def test_audit_metadata_keys_filtered(isolated_rules):
    """Keys prefixed with _ never reach the schema (filtered by load_cio_overrides).
    A future test would re-grow this surface if anyone removes the prefix filter."""
    co, rules_path = isolated_rules
    _write_rules(rules_path, 'trend', {
        '_applied_at': '2026-05-13T00:00:00+00:00',
        '_study': 'macro_v1',
        '_best_value': 0.6,
        'n_estimators': 800,
    })
    merged, applied = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    assert '_applied_at' not in applied
    assert '_study' not in applied
    assert '_best_value' not in applied
    assert applied == {'n_estimators': 800}


def test_unknown_model_returns_defaults(isolated_rules):
    co, rules_path = isolated_rules
    _write_rules(rules_path, 'trend', {'n_estimators': 800})
    # Request a model not in the rules file
    merged, applied = co.merge_with_defaults('does_not_exist', DEFAULTS, SCHEMA)
    assert merged == DEFAULTS
    assert applied == {}


def test_per_model_isolation(isolated_rules):
    """Overrides for one model must not leak into another."""
    co, rules_path = isolated_rules
    rules_path.write_text(json.dumps({
        '_version': 'test',
        'models': {
            'trend':   {'cio_overrides': {'n_estimators': 800}},
            'futures': {'cio_overrides': {'n_estimators': 100}},
        },
    }), encoding='utf-8')

    trend_merged, _ = co.merge_with_defaults('trend', DEFAULTS, SCHEMA)
    fut_merged, _   = co.merge_with_defaults('futures', DEFAULTS, SCHEMA)
    assert trend_merged['n_estimators'] == 800
    assert fut_merged['n_estimators']   == 100


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
