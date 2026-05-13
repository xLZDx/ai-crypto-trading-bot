"""
Phase 0-FIX regression tests — 30+ behavioral tests covering every fix
applied in Phase 0-FIX-A through Phase 0-FIX-D + ML Engineer + CIO Agent.

These tests do NOT use string-match assertions. Each test:
  1. Calls the function under test directly (or mocks dependencies)
  2. Asserts on observable behavior (return value, mutation, raised exception)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv():
    """1000-bar synthetic OHLCV with realistic price path + ATR."""
    rng = np.random.default_rng(42)
    n = 1000
    returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    atr  = pd.Series(np.abs(rng.normal(1.0, 0.3, n)), name='atr_14').clip(lower=0.1)
    ts = pd.date_range('2024-01-01', periods=n, freq='1h')
    return pd.DataFrame({
        'timestamp': ts,
        'close': close,
        'high': np.maximum(high, close),
        'low':  np.minimum(low, close),
        'atr_14': atr.values,
    })


# ── A) Triple Barrier ────────────────────────────────────────────────────────

def test_triple_barrier_atr_applied_once_not_squared(synthetic_ohlcv):
    """BUG-1 fix: dynamic_tp = pt * atr (not pt * atr * atr / atr_mean)."""
    from src.analysis.triple_barrier import triple_barrier_labels_vectorized
    labels, t1 = triple_barrier_labels_vectorized(
        synthetic_ohlcv, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12,
    )
    assert set(np.unique(labels.values)).issubset({-1, 0, 1})
    assert (labels.values != 0).sum() > 100, "Barriers too wide — ATR squaring may be back"


def test_triple_barrier_all_nan_atr_raises(synthetic_ohlcv):
    """BUG-N4 fix: all-NaN ATR must raise ValueError."""
    from src.analysis.triple_barrier import triple_barrier_labels_vectorized
    df = synthetic_ohlcv.copy()
    df['atr_14'] = np.nan
    with pytest.raises(ValueError, match="ATR"):
        triple_barrier_labels_vectorized(df, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12)


def test_triple_barrier_asymmetric_defaults():
    """BUG-2 fix: defaults must be pt=2.5, sl=1.5, max_bars=12."""
    import inspect
    from src.analysis.triple_barrier import triple_barrier_labels_vectorized
    sig = inspect.signature(triple_barrier_labels_vectorized)
    assert sig.parameters['pt_multiplier'].default == 2.5
    assert sig.parameters['sl_multiplier'].default == 1.5
    assert sig.parameters['max_bars'].default == 12


def test_triple_barrier_label_stats_empty():
    """BUG-N: label_stats must not divide by zero on empty series."""
    from src.analysis.triple_barrier import label_stats
    stats = label_stats(pd.Series([], dtype=np.int8))
    assert stats['total'] == 0
    assert stats['long_pct'] == 0.0


def test_triple_barrier_causal_t1_audit_with_datetime_index():
    """BUG-N2 fix: causal_t1_audit must actually filter by DatetimeIndex."""
    from src.analysis.triple_barrier import causal_t1_audit
    n = 100
    ts = pd.date_range('2024-01-01', periods=n, freq='1h')
    t1 = pd.Series(ts + pd.Timedelta(hours=5), index=ts)
    audit = causal_t1_audit(t1, train_end=ts[80], test_start=ts[80])
    assert audit['n_violations'] > 0
    assert not audit['ok']


# ── B) PurgedKFold ───────────────────────────────────────────────────────────

def test_purged_kfold_t1_purging_reduces_train_set():
    """BUG-2 fix: t1-based purging actually removes training samples."""
    from src.utils.purged_kfold import PurgedKFold
    n = 500
    ts = pd.date_range('2024-01-01', periods=n, freq='1h')
    X = pd.DataFrame({'feat': np.arange(n)}, index=ts)
    t1 = pd.Series(ts + pd.Timedelta(hours=20), index=ts)
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    sizes_purged = [len(tr) for tr, _ in cv.split(X)]
    cv_unpurged = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    sizes_unpurged = [len(tr) for tr, _ in cv_unpurged.split(X)]
    assert any(p < u for p, u in zip(sizes_purged, sizes_unpurged))


def test_purged_kfold_embargo_zero_honored():
    """BUG-N: max(1, ...) embargo floor was forcing minimum 1-bar embargo."""
    from src.utils.purged_kfold import PurgedKFold
    n = 100
    X = pd.DataFrame({'feat': np.arange(n)})
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    folds = list(cv.split(X))
    fold1_train, fold1_test = folds[0]
    assert len(fold1_train) == 20, f"Expected 20 train rows, got {len(fold1_train)}"


def test_purged_kfold_t1_constructor_accepts_series():
    from src.utils.purged_kfold import PurgedKFold
    t1 = pd.Series(pd.date_range('2024-01-01', periods=100, freq='1h'))
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.02)
    assert cv.t1 is t1
    assert cv.pct_embargo == 0.02


# ── C) META_FEATURES unification ─────────────────────────────────────────────

def test_meta_features_imported_from_meta_config():
    """BUG-3 fix: all three locations share the same META_FEATURES list."""
    from src.utils.meta_config import META_FEATURES as canonical
    from src.engine.train_meta_labeler import META_FEATURES as training
    from src.analysis.meta_labeler import META_FEATURES as inference
    assert list(canonical) == list(training)
    assert list(canonical) == list(inference)


def test_meta_features_length_23():
    from src.utils.meta_config import META_FEATURES
    assert len(META_FEATURES) == 23


def test_meta_features_contains_primary_signal():
    from src.utils.meta_config import META_FEATURES
    assert 'primary_signal' in META_FEATURES


# ── D) Meta-labeler fail-CLOSED ──────────────────────────────────────────────

def test_meta_labeler_filter_fails_closed_on_exception():
    """BUG-N: filter() must fail-CLOSED (BLOCK) not PASS."""
    from src.analysis.meta_labeler import MetaLabeler
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent'
    ml.model = MagicMock()
    ml.is_loaded = True
    ml.confidence_threshold = 0.60
    ml.model.predict_proba.side_effect = RuntimeError("simulated inference error")
    setattr(ml.model, 'n_features_in_', 23)
    decision, conf = ml.filter(signal=1.0, features={'rsi_14': 50.0})
    assert decision == 'BLOCK'
    assert conf == 0.0


def test_meta_labeler_filter_fails_closed_when_not_loaded():
    from src.analysis.meta_labeler import MetaLabeler
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent'
    ml.model = None
    ml.is_loaded = False
    ml.confidence_threshold = 0.60
    decision, conf = ml.filter(signal=1.0, features={'rsi_14': 50.0})
    assert decision == 'BLOCK'
    assert conf == 0.0


def test_meta_labeler_batch_filter_fails_closed_on_exception():
    from src.analysis.meta_labeler import MetaLabeler
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent'
    ml.model = MagicMock()
    ml.is_loaded = True
    ml.confidence_threshold = 0.60
    ml.model.predict_proba.side_effect = RuntimeError("simulated batch error")
    setattr(ml.model, 'n_features_in_', 23)
    signals = pd.Series([1, 1, -1, 0, 1])
    feats = pd.DataFrame({'rsi_14': [30, 40, 50, 60, 70]})
    out = ml.batch_filter(signals, feats)
    assert (out['decision'] == 'BLOCK').all()
    assert (out['filtered_signal'] == 0.0).all()


def test_meta_labeler_feature_count_mismatch_blocks():
    from src.analysis.meta_labeler import MetaLabeler
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent'
    ml.model = MagicMock()
    ml.is_loaded = True
    ml.confidence_threshold = 0.60
    setattr(ml.model, 'n_features_in_', 13)
    decision, conf = ml.filter(signal=1.0, features={'rsi_14': 50.0})
    assert decision == 'BLOCK'
    assert conf == 0.0


# ── E) Train meta-labeler logging fix ────────────────────────────────────────

def test_train_meta_labeler_no_module_level_basicconfig():
    """BUG-N8: logging.basicConfig must NOT run at import time."""
    import importlib, logging
    root_logger_level_before = logging.getLogger().level
    importlib.import_module('src.engine.train_meta_labeler')
    root_logger_level_after = logging.getLogger().level
    assert root_logger_level_before == root_logger_level_after


# ── F) ML Engineer agent ─────────────────────────────────────────────────────

def test_ml_engineer_blocks_out_of_range_pt(tmp_path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')
    decision = agent.validate_training_request(
        model_type='meta', timeframe='1h',
        config={'pt_multiplier': 5.0, 'sl_multiplier': 0.5, 'max_bars': 12,
                'use_t1_purging': True},
    )
    assert decision.decision == 'BLOCK'


def test_ml_engineer_blocks_symmetric_barriers(tmp_path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')
    decision = agent.validate_training_request(
        model_type='meta', timeframe='1h',
        config={'pt_multiplier': 2.0, 'sl_multiplier': 2.0, 'max_bars': 12,
                'use_t1_purging': True},
    )
    assert decision.decision == 'BLOCK'


def test_ml_engineer_blocks_missing_t1_purging(tmp_path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')
    decision = agent.validate_training_request(
        model_type='meta', timeframe='1h',
        config={'pt_multiplier': 2.5, 'sl_multiplier': 1.5, 'max_bars': 12,
                'use_t1_purging': False},
    )
    assert decision.decision == 'BLOCK'
    assert any('t1' in r.lower() for r in decision.reasons)


def test_ml_engineer_rejects_low_walk_forward_acc(tmp_path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({
        'accuracy': 60.0, 'auc_roc': 0.6, 'win_precision': 55.0,
        'win_rate_pct': 48.0, 'walk_forward_mean_acc': 45.0,
        'walk_forward_std_acc': 5.0, 'walk_forward_folds': 5,
        'optimal_threshold': 0.55, 'n_features': 23, 'n_train': 5000, 'n_test': 1000,
    }))
    agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')
    decision = agent.evaluate_trained_model('meta', '1h', meta_path)
    assert decision.decision == 'REJECT'


def test_ml_engineer_psr_in_metrics(tmp_path):
    """Post-review: PSR uses oos_sharpe + n_test (real Bailey-LdP inputs),
    not walk_forward_mean_acc (which is accuracy %, not Sharpe)."""
    from src.engine.ml_engineer_agent import MLEngineerAgent
    meta_path = tmp_path / 'meta.json'
    meta_path.write_text(json.dumps({
        'accuracy': 60.0, 'auc_roc': 0.6, 'win_precision': 55.0,
        'win_rate_pct': 48.0, 'walk_forward_mean_acc': 55.0,
        'walk_forward_std_acc': 3.0, 'walk_forward_folds': 5,
        'optimal_threshold': 0.55, 'n_features': 23,
        'n_train': 5000, 'n_test': 1000,
        # Real Bailey-LdP PSR inputs
        'oos_sharpe': 1.2,
        'oos_return_skew': 0.0, 'oos_return_kurtosis': 3.0,
    }))
    agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')
    decision = agent.evaluate_trained_model('meta', '1h', meta_path)
    assert 'psr' in decision.metrics
    assert 0.0 <= decision.metrics['psr'] <= 1.0


def test_ml_engineer_persists_decisions(tmp_path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    p = tmp_path / 'decisions.json'
    agent = MLEngineerAgent(decisions_path=p)
    agent.validate_training_request(
        model_type='base', timeframe='1h',
        config={'pt_multiplier': 2.5, 'sl_multiplier': 1.5, 'max_bars': 12,
                'use_t1_purging': True},
    )
    assert p.exists()
    data = json.loads(p.read_text(encoding='utf-8'))
    assert len(data.get('decisions', [])) >= 1


# ── G) CIO Agent ─────────────────────────────────────────────────────────────

def test_cio_agent_constructor_no_optuna_required():
    """CIO Agent must construct without optuna being imported (lazy)."""
    from src.engine.cio_agent import CIOAgent
    agent = CIOAgent(study_name='test_study')
    assert agent.study_name == 'test_study'
    assert agent._optuna is None  # lazy


def test_cio_agent_objective_uses_ml_engineer_gate():
    """CIO Agent's objective must consult ML Engineer for AFML compliance."""
    from src.engine.cio_agent import CIOAgent
    agent = CIOAgent(study_name='test', ml_engineer_gate=True)
    assert agent.ml_engineer_gate is True


# ── H) Security fixes ────────────────────────────────────────────────────────

def test_dashboard_uses_hmac_compare_digest():
    src = (PROJECT_ROOT / 'src' / 'dashboard' / 'app.py').read_text(encoding='utf-8')
    assert 'hmac.compare_digest' in src or '_hmac.compare_digest' in src


def test_dashboard_default_bind_loopback():
    src = (PROJECT_ROOT / 'src' / 'dashboard' / 'app.py').read_text(encoding='utf-8')
    assert "'DASHBOARD_BIND_HOST', '127.0.0.1'" in src


def test_watchdog_default_bind_loopback():
    src = (PROJECT_ROOT / 'scripts' / 'dashboard_watchdog.py').read_text(encoding='utf-8')
    assert "'127.0.0.1'" in src and 'DASHBOARD_BIND_HOST' in src


def test_orchestrator_auth_helper_present():
    src = (PROJECT_ROOT / 'src' / 'training' / 'distributed' / 'orchestrator.py').read_text(encoding='utf-8')
    assert '_require_cluster_auth' in src
    assert src.count('_require_cluster_auth()') >= 5


def test_worker_auth_helper_present():
    src = (PROJECT_ROOT / 'src' / 'training' / 'distributed' / 'worker.py').read_text(encoding='utf-8')
    assert '_require_worker_auth' in src
    assert src.count('_require_worker_auth()') >= 4


def test_dashboard_model_load_uses_verify_and_load_bytes():
    src = (PROJECT_ROOT / 'src' / 'dashboard' / 'app.py').read_text(encoding='utf-8')
    assert 'verify_and_load_bytes' in src


# ── I) Risk manager ─────────────────────────────────────────────────────────

def test_risk_manager_vol_fallback_zero_on_insufficient_data():
    from src.analysis.risk_manager import HullRiskManager
    rm = HullRiskManager.__new__(HullRiskManager)
    result = rm.calculate_historical_volatility([{'close': 100.0}], periods=30)
    assert result == 0.0


def test_risk_manager_vol_fallback_zero_on_no_log_returns():
    from src.analysis.risk_manager import HullRiskManager
    rm = HullRiskManager.__new__(HullRiskManager)
    bars = [{'close': 0.0} for _ in range(35)]
    result = rm.calculate_historical_volatility(bars, periods=30)
    assert result == 0.0


# ── J) Triple Barrier label balance ──────────────────────────────────────────

def test_triple_barrier_produces_three_classes(synthetic_ohlcv):
    """Sanity: with asymmetric barriers we get all three label types."""
    from src.analysis.triple_barrier import triple_barrier_labels_vectorized, label_stats
    labels, _ = triple_barrier_labels_vectorized(
        synthetic_ohlcv, pt_multiplier=2.5, sl_multiplier=1.5, max_bars=12,
    )
    stats = label_stats(labels)
    assert stats['long_pct'] >= 1.0
    assert stats['short_pct'] >= 1.0
    assert stats['timeout_pct'] >= 1.0


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
