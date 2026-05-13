"""
Behavioral tests for src/risk/validators.py — Sprint 0 §S0-1.

Covers:
  - ValidationGate.run() returns APPROVE when all checks pass
  - Disabled config short-circuits with a WARN
  - Stale data triggers BLOCK
  - Label imbalance (single class, minority < min_class_pct) triggers BLOCK
  - High NaN density triggers BLOCK
  - Distribution drift triggers WARN (does not BLOCK)
  - Reports are persisted to validation_runs.json
"""
from __future__ import annotations

import json
import sys
import gzip
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_validators(tmp_path, monkeypatch):
    """Repoint validators paths to tmp + reset singleton."""
    from src.risk import validators as vmod
    monkeypatch.setattr(vmod, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(vmod, 'VALIDATION_LOG_PATH', tmp_path / 'data' / 'risk' / 'validation_runs.json')
    (tmp_path / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    vmod.reset_singleton_for_tests()
    return vmod


def _write_fresh_csv_gz(path: Path, n_bars: int = 200):
    """Helper: write a gzipped CSV with a `timestamp` column ending now."""
    now = datetime.now(timezone.utc)
    rows = [
        {'timestamp': (now - timedelta(hours=n_bars - i - 1)).isoformat(),
         'close': 100.0 + i}
        for i in range(n_bars)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, compression='gzip')


def _write_stale_csv_gz(path: Path, n_bars: int = 200, days_old: int = 7):
    now = datetime.now(timezone.utc) - timedelta(days=days_old)
    rows = [
        {'timestamp': (now - timedelta(hours=n_bars - i - 1)).isoformat(),
         'close': 100.0 + i}
        for i in range(n_bars)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, compression='gzip')


def test_default_run_approve_with_no_data(isolated_validators):
    """No data files = no freshness check triggers = APPROVE (vacuous case)."""
    vmod = isolated_validators
    g = vmod.ValidationGate()
    r = g.run(model_type='meta', timeframe='1h', symbols=[])
    assert r.decision == 'APPROVE'
    assert r.reasons == []


def test_disabled_config_returns_warn(isolated_validators):
    vmod = isolated_validators
    cfg = vmod.ValidationConfig(enabled=False)
    g = vmod.ValidationGate(cfg=cfg)
    r = g.run(model_type='base', timeframe='1h')
    # decision stays APPROVE but with a warning entry
    assert any('disabled' in w.lower() for w in r.warnings)


def test_stale_data_triggers_block(isolated_validators, tmp_path):
    vmod = isolated_validators
    raw = tmp_path / 'data' / 'raw'
    _write_stale_csv_gz(raw / 'BTC_USDT_1h.csv.gz', days_old=10)
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', symbols=['BTC_USDT'])
    assert r.decision == 'BLOCK'
    assert any('stale' in reason for reason in r.reasons)
    assert 'BTC_USDT' in r.metrics.get('stale_symbols', [])


def test_fresh_data_does_not_block(isolated_validators, tmp_path):
    vmod = isolated_validators
    raw = tmp_path / 'data' / 'raw'
    _write_fresh_csv_gz(raw / 'BTC_USDT_1h.csv.gz')
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', symbols=['BTC_USDT'])
    assert r.decision == 'APPROVE'


def test_label_imbalance_blocks_when_single_class(isolated_validators):
    vmod = isolated_validators
    labels = pd.Series([1] * 100, dtype=np.int8)
    g = vmod.ValidationGate()
    r = g.run(model_type='meta', timeframe='1h', labels=labels)
    assert r.decision == 'BLOCK'
    assert any('label imbalance' in reason.lower() for reason in r.reasons)


def test_label_imbalance_blocks_when_minority_under_threshold(isolated_validators):
    vmod = isolated_validators
    # 95% class 1, 5% class 0 — minority below default 10%
    labels = pd.Series([1] * 95 + [0] * 5, dtype=np.int8)
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', labels=labels)
    assert r.decision == 'BLOCK'


def test_label_balance_above_threshold_passes(isolated_validators):
    vmod = isolated_validators
    labels = pd.Series([1] * 60 + [0] * 40, dtype=np.int8)
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', labels=labels)
    assert r.decision == 'APPROVE'


def test_high_nan_density_blocks(isolated_validators):
    vmod = isolated_validators
    df = pd.DataFrame({'a': [np.nan] * 50 + [1.0] * 50,
                       'b': [np.nan] * 80 + [1.0] * 20})  # 65% NaN
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', feature_df=df)
    assert r.decision == 'BLOCK'
    assert r.metrics['nan_pct'] > 0.5


def test_low_nan_density_passes(isolated_validators):
    vmod = isolated_validators
    df = pd.DataFrame({'a': [1.0] * 100, 'b': [2.0] * 100})
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h', feature_df=df)
    assert r.decision == 'APPROVE'
    assert r.metrics['nan_pct'] == 0.0


def test_distribution_drift_emits_warn_not_block(isolated_validators):
    vmod = isolated_validators
    # Feature with mean ≈ 100 in production
    df = pd.DataFrame({'rsi_14': np.random.normal(100.0, 1.0, 200)})
    # Baseline says mean = 50 — strong drift
    baseline = {'rsi_14': {'mean': 50.0, 'std': 5.0}}
    g = vmod.ValidationGate()
    r = g.run(model_type='base', timeframe='1h',
              feature_df=df, last_known_good_dist=baseline)
    assert r.decision == 'WARN'
    assert 'rsi_14' in r.metrics.get('drifted_features', [])


def test_persists_runs(isolated_validators):
    vmod = isolated_validators
    g = vmod.ValidationGate()
    g.run(model_type='base', timeframe='1h')
    assert vmod.VALIDATION_LOG_PATH.exists()
    data = json.loads(vmod.VALIDATION_LOG_PATH.read_text())
    assert len(data.get('runs', [])) == 1


def test_singleton_consistency(isolated_validators):
    a = isolated_validators.get_validation_gate()
    b = isolated_validators.get_validation_gate()
    assert a is b


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
