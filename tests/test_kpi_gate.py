"""
Behavioral tests for src/engine/kpi_gate.py (Sprint 1A R2).

Covers:
  - TrainingResult.successful property
  - append_run + last_n_successful round-trip via per-cell Parquet
  - thresholds_for() reading from training_rules.json
  - _check_thresholds: MIN for normal fields, MAX for wf_max_dd
  - evaluate_run: APPROVE / strike-1 / strike-2 / RETIRE on 3rd consecutive miss
  - is_retired / restore / persistent registry behavior
  - evaluate_from_meta_json: orchestrator-facing entrypoint
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def kpi_gate_isolated(tmp_path, monkeypatch):
    """Re-bind the module-level path constants to tmp_path so tests don't
    touch real data/ files. Returns the module."""
    import importlib
    from src.engine import kpi_gate as kg
    monkeypatch.setattr(kg, 'TRAINING_RUNS_DIR', tmp_path / 'training_runs')
    monkeypatch.setattr(kg, 'RETIRED_REGISTRY_PATH', tmp_path / 'kpi_retired.json')
    # Use a tmp training_rules.json with known thresholds
    rules = {
        "models": {
            "base": {"kpi_threshold": {"wf_acc": 50.0, "wf_total_trades": 30}},
            "trend": {"kpi_threshold": {"wf_acc": 50.0, "wf_max_dd": 0.20}},
            "no_threshold_model": {"applicable_tfs": ["1h"]},  # no kpi_threshold
        }
    }
    rules_path = tmp_path / 'training_rules.json'
    rules_path.write_text(json.dumps(rules))
    monkeypatch.setattr(kg, 'TRAINING_RULES_PATH', rules_path)
    return kg


def _ok_result(kg, model='base', tf='1h', wf_acc=55.0, wf_total_trades=100, **extra):
    """Build a TrainingResult that passes default thresholds."""
    return kg.TrainingResult(
        model_key=model, tf=tf,
        started_at=time.time() - 60, finished_at=time.time(),
        artifact_path=f'/fake/{model}_{tf}.joblib',
        n_samples=10000, n_features=20,
        wf_acc=wf_acc, wf_total_trades=wf_total_trades,
        **extra,
    )


def _bad_result(kg, model='base', tf='1h', wf_acc=45.0, wf_total_trades=10):
    """Build a TrainingResult that FAILS thresholds (wf_acc<50, wf_total_trades<30)."""
    return kg.TrainingResult(
        model_key=model, tf=tf,
        started_at=time.time() - 60, finished_at=time.time(),
        artifact_path=f'/fake/{model}_{tf}.joblib',
        n_samples=10000, n_features=20,
        wf_acc=wf_acc, wf_total_trades=wf_total_trades,
    )


# ── TrainingResult ───────────────────────────────────────────────────────────

def test_training_result_successful_true(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = _ok_result(kg)
    assert r.successful is True


def test_training_result_successful_false_on_error(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = kg.TrainingResult(
        model_key='base', tf='1h',
        started_at=0, finished_at=1, artifact_path=None,
        error='train failed',
    )
    assert r.successful is False


def test_training_result_successful_false_when_cancelled(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = kg.TrainingResult(
        model_key='base', tf='1h',
        started_at=0, finished_at=1, artifact_path='/fake.joblib',
        cancelled=True,
    )
    assert r.successful is False


# ── append_run + last_n_successful round-trip ────────────────────────────────

def test_append_then_last_n(kpi_gate_isolated):
    kg = kpi_gate_isolated
    for i in range(5):
        r = _ok_result(kg, wf_acc=55.0 + i)
        # Stagger finished_at so order matters
        r.finished_at = float(1_000_000 + i)
        kg.append_run(r)
    recent = kg.last_n_successful('base', '1h', n=3)
    assert len(recent) == 3
    # Most recent first
    assert recent[0].finished_at == 1_000_004
    assert recent[1].finished_at == 1_000_003
    assert recent[2].finished_at == 1_000_002


def test_last_n_excludes_failed_runs(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # 2 successful + 1 failed
    kg.append_run(_ok_result(kg))
    bad = kg.TrainingResult(
        model_key='base', tf='1h',
        started_at=0, finished_at=time.time(),
        artifact_path=None, error='boom',
    )
    kg.append_run(bad)
    kg.append_run(_ok_result(kg))
    recent = kg.last_n_successful('base', '1h', n=10)
    assert len(recent) == 2  # failed run excluded


def test_last_n_empty_for_unknown_cell(kpi_gate_isolated):
    kg = kpi_gate_isolated
    assert kg.last_n_successful('does_not_exist', '1h') == []


# ── thresholds_for ───────────────────────────────────────────────────────────

def test_thresholds_for_known_model(kpi_gate_isolated):
    kg = kpi_gate_isolated
    thr = kg.thresholds_for('base')
    assert thr == {'wf_acc': 50.0, 'wf_total_trades': 30}


def test_thresholds_for_model_without_threshold(kpi_gate_isolated):
    kg = kpi_gate_isolated
    assert kg.thresholds_for('no_threshold_model') is None


def test_thresholds_for_unknown_model(kpi_gate_isolated):
    kg = kpi_gate_isolated
    assert kg.thresholds_for('totally_unknown') is None


# ── _check_thresholds: MIN vs MAX semantics ──────────────────────────────────

def test_check_thresholds_pass_when_above_min(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = _ok_result(kg, wf_acc=60.0, wf_total_trades=100)
    missed = kg._check_thresholds(r, {'wf_acc': 50.0, 'wf_total_trades': 30})
    assert missed == []


def test_check_thresholds_fail_when_below_min(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = _ok_result(kg, wf_acc=45.0)
    missed = kg._check_thresholds(r, {'wf_acc': 50.0})
    assert len(missed) == 1
    assert 'wf_acc' in missed[0]


def test_check_thresholds_max_dd_inverted(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # wf_max_dd is a MAX threshold (smaller is better)
    r = _ok_result(kg, wf_max_dd=0.10)  # 10% drawdown
    assert kg._check_thresholds(r, {'wf_max_dd': 0.20}) == []  # pass: 0.10 <= 0.20
    r2 = _ok_result(kg, wf_max_dd=0.25)
    miss = kg._check_thresholds(r2, {'wf_max_dd': 0.20})
    assert len(miss) == 1 and 'wf_max_dd' in miss[0]  # fail: 0.25 > 0.20


def test_check_thresholds_missing_field(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = kg.TrainingResult(
        model_key='base', tf='1h',
        started_at=0, finished_at=time.time(),
        artifact_path='/fake.joblib',
        # wf_acc deliberately left None
    )
    miss = kg._check_thresholds(r, {'wf_acc': 50.0})
    assert any('missing' in m for m in miss)


# ── evaluate_run: full lifecycle ─────────────────────────────────────────────

def test_evaluate_run_strike_1_no_retire(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # First failing run, no prior history
    outcome = kg.evaluate_run(_bad_result(kg))
    assert outcome['persisted'] is True
    assert outcome['retired_now'] is False
    # Need at least CONSECUTIVE_MISS_LIMIT runs before evaluation can retire
    assert 'only' in outcome['reasons'][0]  # "only 1/3 runs available"


def test_evaluate_run_3_consecutive_misses_retires(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # All 3 runs fail thresholds → retire on the 3rd
    for i in range(2):
        r = _bad_result(kg)
        r.finished_at = float(1000 + i)
        kg.evaluate_run(r)
        assert kg.is_retired('base', '1h') is False
    # Third strike
    r3 = _bad_result(kg)
    r3.finished_at = 1003.0
    final = kg.evaluate_run(r3)
    assert final['retired_now'] is True
    assert kg.is_retired('base', '1h') is True
    assert len(final['missed_fields']) >= 1


def test_evaluate_run_pass_resets_strike_counter(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # 2 failing, then 1 passing → should NOT retire (last-3 must all fail)
    for i in range(2):
        r = _bad_result(kg)
        r.finished_at = float(1000 + i)
        kg.evaluate_run(r)
    r_pass = _ok_result(kg)
    r_pass.finished_at = 1002.0
    outcome = kg.evaluate_run(r_pass)
    assert outcome['retired_now'] is False
    assert kg.is_retired('base', '1h') is False


def test_evaluate_run_no_threshold_for_model(kpi_gate_isolated):
    kg = kpi_gate_isolated
    r = _ok_result(kg, model='no_threshold_model')
    outcome = kg.evaluate_run(r)
    assert outcome['retired_now'] is False
    assert any('no kpi_threshold' in reason for reason in outcome['reasons'])


def test_evaluate_run_failed_training_not_evaluated(kpi_gate_isolated):
    kg = kpi_gate_isolated
    bad = kg.TrainingResult(
        model_key='base', tf='1h',
        started_at=0, finished_at=time.time(),
        artifact_path=None, error='failure',
    )
    outcome = kg.evaluate_run(bad)
    assert outcome['retired_now'] is False
    assert any('run failed' in r for r in outcome['reasons'])


# ── restore + persistent registry ────────────────────────────────────────────

def test_restore_clears_retired_flag(kpi_gate_isolated):
    kg = kpi_gate_isolated
    # Force-retire then restore
    for i in range(3):
        r = _bad_result(kg)
        r.finished_at = float(1000 + i)
        kg.evaluate_run(r)
    assert kg.is_retired('base', '1h') is True
    result = kg.restore('base', '1h')
    assert result['ok'] is True
    assert kg.is_retired('base', '1h') is False


def test_restore_unknown_cell_returns_not_ok(kpi_gate_isolated):
    kg = kpi_gate_isolated
    result = kg.restore('does_not_exist', '1h')
    assert result['ok'] is False


# ── evaluate_from_meta_json: orchestrator entrypoint ─────────────────────────

def test_evaluate_from_meta_json_happy_path(kpi_gate_isolated, tmp_path):
    kg = kpi_gate_isolated
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({
        'artifact_path': '/fake/base_1h.joblib',
        'n_samples': 10000, 'n_features': 33,
        'walk_forward_mean_acc': 55.0,
        'walk_forward_total_trades': 100,
        'finished_at_ts': time.time(),
    }))
    outcome = kg.evaluate_from_meta_json('base', '1h', str(meta_path))
    assert outcome['persisted'] is True


def test_evaluate_from_meta_json_unreadable_meta(kpi_gate_isolated, tmp_path):
    kg = kpi_gate_isolated
    outcome = kg.evaluate_from_meta_json('base', '1h', str(tmp_path / 'does_not_exist.json'))
    assert outcome['persisted'] is False
    assert any('unreadable' in r for r in outcome['reasons'])


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
