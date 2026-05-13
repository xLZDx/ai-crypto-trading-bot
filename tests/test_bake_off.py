"""
Behavioral tests for src/engine/bake_off.py — Sprint 0 §S0-2.

Covers:
  - run_bake_off enumerates cells from training_rules.json
  - Cells with no successful runs are recommended 'review'
  - Cells already retired by KPI gate are recommended 'retire'
  - Bottom retire_below_pct of populated cells are recommended 'retire'
  - INVERTED_METRICS (wf_max_dd) sorts ascending (smaller is better)
  - Cut list persisted to data/bake_off_cut_list.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_bake_off(tmp_path, monkeypatch):
    """Repoint bake_off + kpi_gate paths to tmp."""
    from src.engine import bake_off as bo
    monkeypatch.setattr(bo, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(bo, 'TRAINING_RUNS_DIR', tmp_path / 'data' / 'training_runs')
    monkeypatch.setattr(bo, 'CUT_LIST_PATH', tmp_path / 'data' / 'bake_off_cut_list.json')
    rules_dir = tmp_path / 'data'
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / 'training_rules.json').write_text(json.dumps({
        'models': {
            'meta':     {'applicable_tfs': ['1h'], 'experimental_tfs': []},
            'base':     {'applicable_tfs': ['1h', '4h'], 'experimental_tfs': []},
            'scalping': {'applicable_tfs': ['1m'], 'experimental_tfs': []},
        }
    }))
    return bo


def _mock_kg_run(model: str, tf: str, wf_sharpe: float | None,
                 wf_max_dd: float | None = None):
    """Return a fake TrainingResult-shaped object for KPI gate stubbing."""
    m = MagicMock()
    m.wf_sharpe       = wf_sharpe
    m.wf_calmar       = None
    m.wf_max_dd       = wf_max_dd
    m.wf_win_rate     = None
    m.wf_expectancy   = None
    m.wf_total_trades = None
    m.wf_acc          = None
    m.auc_roc         = None
    m.n_samples       = 1000
    m.n_features      = 20
    m.finished_at     = 1234567890.0
    return m


def test_no_runs_all_review(isolated_bake_off):
    """When no KPI runs exist, every cell is 'review' (no metric)."""
    bo = isolated_bake_off
    with patch('src.engine.kpi_gate.last_n_successful', return_value=[]), \
         patch('src.engine.kpi_gate.is_retired', return_value=False):
        out = bo.run_bake_off(metric='wf_sharpe', top_n=10)
    assert len(out['cut_list']['review']) == 4  # meta/1h, base/1h, base/4h, scalping/1m
    assert out['cut_list']['retire'] == []
    assert out['cut_list']['keep'] == []


def test_retired_by_kpi_gate_marked_retire(isolated_bake_off):
    bo = isolated_bake_off
    with patch('src.engine.kpi_gate.last_n_successful', return_value=[]), \
         patch('src.engine.kpi_gate.is_retired',
               side_effect=lambda m, t: m == 'scalping' and t == '1m'):
        out = bo.run_bake_off(metric='wf_sharpe', top_n=10)
    assert 'scalping__1m' in out['cut_list']['retire']


def test_bottom_pct_marked_retire(isolated_bake_off):
    """With 4 populated cells, retire_below_pct=0.25 → bottom 1 cell retires."""
    bo = isolated_bake_off
    # Stub: 4 cells, decreasing sharpe
    sharpes = {
        ('meta', '1h'):     [_mock_kg_run('meta', '1h', 2.5)],
        ('base', '1h'):     [_mock_kg_run('base', '1h', 1.5)],
        ('base', '4h'):     [_mock_kg_run('base', '4h', 1.0)],
        ('scalping', '1m'): [_mock_kg_run('scalping', '1m', 0.5)],
    }
    def fake_runs(model, tf, n=1):
        return sharpes.get((model, tf), [])
    with patch('src.engine.kpi_gate.last_n_successful', side_effect=fake_runs), \
         patch('src.engine.kpi_gate.is_retired', return_value=False):
        out = bo.run_bake_off(metric='wf_sharpe', top_n=10, retire_below_pct=0.25)
    # Worst (scalping__1m, sharpe=0.5) is in bottom 25% → retire
    assert 'scalping__1m' in out['cut_list']['retire']
    # Top 3 should be keep
    assert 'meta__1h' in out['cut_list']['keep']
    assert 'base__1h' in out['cut_list']['keep']


def test_inverted_metric_max_dd_smaller_is_better(isolated_bake_off):
    """wf_max_dd: 0.05 (small drawdown) ranks BETTER than 0.30 (big drawdown)."""
    bo = isolated_bake_off
    dds = {
        ('meta', '1h'):     [_mock_kg_run('meta', '1h', wf_sharpe=None, wf_max_dd=0.05)],
        ('base', '1h'):     [_mock_kg_run('base', '1h', wf_sharpe=None, wf_max_dd=0.30)],
        ('base', '4h'):     [_mock_kg_run('base', '4h', wf_sharpe=None, wf_max_dd=0.15)],
        ('scalping', '1m'): [_mock_kg_run('scalping', '1m', wf_sharpe=None, wf_max_dd=0.20)],
    }
    def fake_runs(model, tf, n=1):
        return dds.get((model, tf), [])
    with patch('src.engine.kpi_gate.last_n_successful', side_effect=fake_runs), \
         patch('src.engine.kpi_gate.is_retired', return_value=False):
        out = bo.run_bake_off(metric='wf_max_dd', top_n=10, retire_below_pct=0.25)
    # Best ranked first ⇒ rows[0] should be the smallest max_dd
    assert out['rows'][0]['model'] == 'meta'
    assert out['rows'][0]['metric_value'] == 0.05
    # Worst (base/1h at 0.30) is in bottom 25% → retire
    assert 'base__1h' in out['cut_list']['retire']


def test_cut_list_persisted(isolated_bake_off):
    bo = isolated_bake_off
    with patch('src.engine.kpi_gate.last_n_successful', return_value=[]), \
         patch('src.engine.kpi_gate.is_retired', return_value=False):
        bo.run_bake_off(metric='wf_sharpe', top_n=10)
    assert bo.CUT_LIST_PATH.exists()
    data = json.loads(bo.CUT_LIST_PATH.read_text())
    assert 'rows' in data
    assert 'cut_list' in data
    assert 'generated_at' in data


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
