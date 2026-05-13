"""
Tests for src/risk/drift_baseline.py — Sprint 0 §S0-1 follow-on.

Covers:
  - save_baseline: empty df → empty dict (no file)
  - save_baseline: produces per-feature mean/std/q05/q95/n
  - load_baseline: returns dict when file is fresh
  - load_baseline: returns None when file is older than max_age_days
  - load_baseline: returns None when file is missing
  - load_baseline: handles malformed JSON gracefully
  - baseline_age_days returns a positive float for an existing file
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_baselines(tmp_path, monkeypatch):
    from src.risk import drift_baseline as db
    monkeypatch.setattr(db, 'BASELINES_DIR', tmp_path)
    return db, tmp_path


def test_save_baseline_empty_returns_empty(isolated_baselines):
    db, _ = isolated_baselines
    out = db.save_baseline('meta', '1h', pd.DataFrame())
    assert out == {}


def test_save_baseline_persists_per_feature_stats(isolated_baselines):
    db, tmp = isolated_baselines
    df = pd.DataFrame({
        'rsi_14': np.random.normal(50.0, 10.0, 200),
        'atr_14': np.random.normal(1.5, 0.3, 200),
    })
    out = db.save_baseline('meta', '1h', df)
    assert out['feature_count'] == 2
    assert 'rsi_14' in out['features']
    assert 'mean' in out['features']['rsi_14']
    assert 'std'  in out['features']['rsi_14']
    assert (tmp / 'meta__1h.json').exists()


def test_load_baseline_returns_fresh_payload(isolated_baselines):
    db, _ = isolated_baselines
    df = pd.DataFrame({'rsi_14': np.random.normal(50.0, 10.0, 100)})
    db.save_baseline('meta', '1h', df)
    out = db.load_baseline('meta', '1h')
    assert out is not None
    assert 'rsi_14' in out
    assert isinstance(out['rsi_14']['mean'], float)


def test_load_baseline_missing_returns_none(isolated_baselines):
    db, _ = isolated_baselines
    assert db.load_baseline('does_not_exist', '1h') is None


def test_load_baseline_stale_returns_none(isolated_baselines):
    db, tmp = isolated_baselines
    # Write a baseline with an old timestamp
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    payload = {
        'model_key': 'meta', 'timeframe': '1h',
        'saved_at': stale_ts,
        'features': {'rsi_14': {'mean': 50.0, 'std': 5.0, 'q05': 40, 'q95': 60, 'n': 100}},
    }
    (tmp / 'meta__1h.json').write_text(json.dumps(payload))
    assert db.load_baseline('meta', '1h', max_age_days=30) is None


def test_load_baseline_malformed_returns_none(isolated_baselines):
    db, tmp = isolated_baselines
    (tmp / 'meta__1h.json').write_text('{this is not json')
    assert db.load_baseline('meta', '1h') is None


def test_baseline_age_days(isolated_baselines):
    db, _ = isolated_baselines
    df = pd.DataFrame({'col': np.arange(100, dtype=float)})
    db.save_baseline('meta', '1h', df)
    age = db.baseline_age_days('meta', '1h')
    assert age is not None
    assert age < 0.1  # just saved → very young


def test_baseline_age_days_missing_returns_none(isolated_baselines):
    db, _ = isolated_baselines
    assert db.baseline_age_days('does_not_exist', '1h') is None


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
