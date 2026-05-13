"""
Tests for the cio_overrides MERGE into train_model._load_model_params.

Verifies:
  - With no cio_overrides, validated params are unchanged from training_rules.params
  - With cio_overrides matching schema, merged values land in validated dict
  - cio_overrides with WRONG TYPE are skipped (NOT merged), default kept
  - cio_overrides OUT OF RANGE are skipped, default kept
  - Audit-only keys (_applied_at etc.) never reach the validated dict
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
    """Repoint both train_model._RULES_PATH AND the shared cio_overrides path."""
    from src.engine import train_model as tm
    from src.utils import cio_overrides as co
    rules_path = tmp_path / 'training_rules.json'
    monkeypatch.setattr(tm, '_RULES_PATH', str(rules_path))
    monkeypatch.setattr(co, 'TRAINING_RULES_PATH', rules_path)
    return tm, rules_path


def _base_rules(cio_overrides=None):
    cell = {
        'params': {
            'n_estimators': 200,
            'max_depth':    6,
            'class_weight': 'balanced',
        }
    }
    if cio_overrides is not None:
        cell['cio_overrides'] = cio_overrides
    return {'_version': 'test_v1', 'models': {'base': cell}}


def test_no_overrides_returns_unchanged(isolated_rules):
    tm, rules_path = isolated_rules
    rules_path.write_text(json.dumps(_base_rules()))
    validated, _ver, _hash = tm._load_model_params('base')
    assert validated == {
        'n_estimators': 200, 'max_depth': 6, 'class_weight': 'balanced',
    }


def test_valid_override_merged(isolated_rules):
    tm, rules_path = isolated_rules
    rules_path.write_text(json.dumps(_base_rules(cio_overrides={
        'n_estimators': 350,
        'max_depth':    10,
        '_applied_at':  '2026-05-13T00:00:00+00:00',
    })))
    validated, _ver, _hash = tm._load_model_params('base')
    # CIO overrides win for the two schema-valid keys
    assert validated['n_estimators'] == 350
    assert validated['max_depth']    == 10
    # Class weight not in override → stays at base value
    assert validated['class_weight'] == 'balanced'


def test_wrong_type_override_skipped(isolated_rules):
    tm, rules_path = isolated_rules
    rules_path.write_text(json.dumps(_base_rules(cio_overrides={
        'n_estimators': 'lots',  # wrong type — should be int
    })))
    validated, _, _ = tm._load_model_params('base')
    # The bad override is skipped → default from training_rules.params used
    assert validated['n_estimators'] == 200


def test_out_of_range_override_skipped(isolated_rules):
    tm, rules_path = isolated_rules
    rules_path.write_text(json.dumps(_base_rules(cio_overrides={
        'max_depth': 999,  # _HP_SCHEMA caps max_depth at 50
    })))
    validated, _, _ = tm._load_model_params('base')
    assert validated['max_depth'] == 6


def test_audit_metadata_never_merged(isolated_rules):
    tm, rules_path = isolated_rules
    rules_path.write_text(json.dumps(_base_rules(cio_overrides={
        '_applied_at': 'should not leak',
        '_study': 'foo',
        '_best_value': 2.7,
    })))
    validated, _, _ = tm._load_model_params('base')
    assert '_applied_at' not in validated
    assert '_study' not in validated


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
