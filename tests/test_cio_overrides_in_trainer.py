"""
Tests for _load_cio_overrides in train_meta_labeler.

Verifies that operator-approved CIO proposals (written by apply_best) reach
the trainer correctly:
  - Empty/missing overrides → returns {}
  - Metadata keys (_applied_at, _study, _best_value) are stripped
  - Plain HP keys flow through
  - Missing rules file → empty dict (graceful degradation)
  - Malformed JSON → empty dict
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
    """Repoint the shared cio_overrides helper to read from tmp.

    Post-refactor (db5504b → 44b8a4f), train_meta_labeler._load_cio_overrides
    delegates to src.utils.cio_overrides.load_cio_overrides which uses its
    own TRAINING_RULES_PATH constant. Patch THAT, not the trainer's path.
    """
    from src.engine import train_meta_labeler as tm
    from src.utils import cio_overrides as co
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    rules_path = tmp_path / 'data' / 'training_rules.json'
    monkeypatch.setattr(co, 'TRAINING_RULES_PATH', rules_path)
    # Some tests reference tmp_path / 'data' / 'training_rules.json' — keep
    # the same layout so they can write the rules file there.
    return tm, tmp_path


def test_no_rules_file_returns_empty(isolated_rules):
    tm, _ = isolated_rules
    assert tm._load_cio_overrides('meta') == {}


def test_no_cio_overrides_returns_empty(isolated_rules):
    tm, tmp = isolated_rules
    (tmp / 'data' / 'training_rules.json').write_text(json.dumps({
        'models': {'meta': {'params': {'n_estimators': 100}}}
    }))
    assert tm._load_cio_overrides('meta') == {}


def test_overrides_with_metadata_stripped(isolated_rules):
    tm, tmp = isolated_rules
    (tmp / 'data' / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta': {
                'cio_overrides': {
                    'pt_multiplier': 2.5,
                    'sl_multiplier': 1.5,
                    'max_bars': 12,
                    'confidence_threshold': 0.54,
                    '_applied_at': '2026-05-13T12:00:00+00:00',
                    '_study': 'macro_parameter_search_v1',
                    '_best_value': 2.7,
                }
            }
        }
    }))
    result = tm._load_cio_overrides('meta')
    assert result == {
        'pt_multiplier': 2.5,
        'sl_multiplier': 1.5,
        'max_bars': 12,
        'confidence_threshold': 0.54,
    }
    # No metadata leakage
    assert all(not k.startswith('_') for k in result)


def test_unknown_model_returns_empty(isolated_rules):
    tm, tmp = isolated_rules
    (tmp / 'data' / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta': {'cio_overrides': {'pt_multiplier': 2.5}}
        }
    }))
    assert tm._load_cio_overrides('not_a_real_model') == {}


def test_malformed_json_returns_empty(isolated_rules):
    tm, tmp = isolated_rules
    (tmp / 'data' / 'training_rules.json').write_text('{this is not json')
    # Must NOT raise — graceful degradation
    assert tm._load_cio_overrides('meta') == {}


def test_overrides_for_non_meta_models(isolated_rules):
    """Same mechanism applies to base/trend/scalping when their trainers
    eventually wire _load_cio_overrides — verify the lookup is keyed."""
    tm, tmp = isolated_rules
    (tmp / 'data' / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta':  {'cio_overrides': {'pt_multiplier': 2.5}},
            'base':  {'cio_overrides': {'n_estimators': 200}},
            'trend': {'cio_overrides': {'max_depth': 6}},
        }
    }))
    assert tm._load_cio_overrides('meta')  == {'pt_multiplier': 2.5}
    assert tm._load_cio_overrides('base')  == {'n_estimators': 200}
    assert tm._load_cio_overrides('trend') == {'max_depth': 6}


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
