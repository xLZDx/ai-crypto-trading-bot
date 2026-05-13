"""
Edge-case behavioral tests for MetaLabeler (src/analysis/meta_labeler.py).

Safety invariant: MetaLabeler is SAFETY-CRITICAL — it must NEVER fail-OPEN.
Every error path must return ('BLOCK', 0.0) or an all-BLOCK DataFrame.

Test methodology:
- All tests inject a MagicMock model via __new__ to bypass _load() entirely.
- Tests call filter() / batch_filter() directly and assert on return values.
- No string-match assertions are used as primary coverage.
- Each test covers exactly one behaviour so failures are pinpointed.

Coverage: 18 edge cases mandated by task specification + 5 bug-regression
          tests surfaced during pre-flight analysis (marked BUG-REGRESSION).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.meta_labeler import MetaLabeler
from src.utils.meta_config import META_FEATURES, CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ml(*, p_win: float = 0.75, n_features_in: int = 23,
             threshold: float = CONFIDENCE_THRESHOLD) -> MetaLabeler:
    """Return a MetaLabeler with a mocked model — bypasses _load() entirely."""
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent/meta_labeler.joblib'
    ml.model = MagicMock()
    ml.is_loaded = True
    ml.confidence_threshold = threshold
    # Two-class probability: (1 - p_win, p_win)
    ml.model.predict_proba.return_value = np.array([[1.0 - p_win, p_win]])
    setattr(ml.model, 'n_features_in_', n_features_in)
    return ml


def _full_features() -> dict:
    """Return a complete feature dict with all 23 META_FEATURES filled."""
    return {f: float(i) for i, f in enumerate(META_FEATURES)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. signal=0.0 — no signal, must BLOCK regardless of model
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_zero_signal_blocks_regardless_of_model():
    """signal=0.0 is below the abs(signal)<0.1 guard — must return ('BLOCK', 0.0)."""
    ml = _make_ml(p_win=0.99)  # model would PASS if called
    decision, conf = ml.filter(signal=0.0, features={'rsi_14': 50.0})
    assert decision == 'BLOCK', "Zero signal must be BLOCKed"
    assert conf == 0.0
    # Model must NOT be called — zero signal is short-circuited before inference
    ml.model.predict_proba.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 2. signal=0.05 — below 0.1 threshold, treated as no-signal
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_near_zero_signal_blocks():
    """signal=0.05: abs(0.05)<0.1 — treated as no signal, must BLOCK."""
    ml = _make_ml(p_win=0.99)
    decision, conf = ml.filter(signal=0.05, features={'rsi_14': 50.0})
    assert decision == 'BLOCK'
    assert conf == 0.0
    ml.model.predict_proba.assert_not_called()


def test_filter_negative_near_zero_signal_blocks():
    """signal=-0.05: abs(-0.05)<0.1 — also treated as no signal."""
    ml = _make_ml(p_win=0.99)
    decision, conf = ml.filter(signal=-0.05, features={'rsi_14': 50.0})
    assert decision == 'BLOCK'
    assert conf == 0.0
    ml.model.predict_proba.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 3. signal=-1.0 — short signal processed same as +1.0 long signal
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_short_signal_processed_identically_to_long():
    """signal=-1.0 and signal=+1.0 must produce identical (decision, confidence)."""
    ml_long = _make_ml(p_win=0.75)
    ml_short = _make_ml(p_win=0.75)
    feats = _full_features()
    dec_long, conf_long = ml_long.filter(signal=1.0, features=feats)
    dec_short, conf_short = ml_short.filter(signal=-1.0, features=feats)
    assert dec_long == dec_short, "Short and long signals must be treated identically"
    assert conf_long == conf_short


def test_filter_short_signal_can_pass():
    """A short signal with high p_win must PASS — not be unconditionally BLOCKed."""
    ml = _make_ml(p_win=0.90, threshold=0.60)
    decision, _ = ml.filter(signal=-1.0, features=_full_features())
    assert decision == 'PASS'


# ─────────────────────────────────────────────────────────────────────────────
# 4. features as pd.Series — must be converted to dict internally
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_accepts_pd_series_features():
    """features passed as pd.Series must be silently converted and score correctly."""
    ml = _make_ml(p_win=0.80, threshold=0.60)
    features_series = pd.Series(_full_features())
    decision, conf = ml.filter(signal=1.0, features=features_series)
    assert decision == 'PASS'
    assert conf == pytest.approx(0.80, abs=1e-4)


def test_filter_series_features_model_receives_correct_shape():
    """Model must receive a DataFrame with exactly 23 columns when given a Series."""
    ml = _make_ml(p_win=0.75)
    features_series = pd.Series(_full_features())
    ml.filter(signal=1.0, features=features_series)
    call_args = ml.model.predict_proba.call_args
    X_passed = call_args[0][0]
    assert X_passed.shape == (1, len(META_FEATURES)), (
        f"Model received X with shape {X_passed.shape}, expected (1, {len(META_FEATURES)})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. features as list[dict] — must use the last dict
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_list_of_dicts_uses_last_element():
    """When features is a list of dicts, only the last dict should be used."""
    ml = _make_ml(p_win=0.75)
    first_dict = {'rsi_14': 10.0}  # would produce different outcome if used
    last_dict = _full_features()
    ml.filter(signal=1.0, features=[first_dict, last_dict])
    call_args = ml.model.predict_proba.call_args
    X_passed = call_args[0][0]
    # The last dict had 'rsi_14' = META_FEATURES.index('rsi_14') = 7.0
    rsi_idx = META_FEATURES.index('rsi_14')
    assert float(X_passed.iloc[0]['rsi_14']) == pytest.approx(float(rsi_idx), abs=1e-6), (
        "filter() must use the LAST dict in the list"
    )


def test_filter_list_single_dict_works():
    """list[dict] with a single entry must work without IndexError."""
    ml = _make_ml(p_win=0.75)
    decision, conf = ml.filter(signal=1.0, features=[_full_features()])
    assert decision in ('PASS', 'BLOCK')
    assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. features as empty dict — neutral priors fill missing, model still scores
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_empty_dict_features_fills_neutral_priors_and_scores(caplog):
    """Empty dict: missing prob_base/prob_trend/regime get neutral defaults; model runs."""
    ml = _make_ml(p_win=0.75, threshold=0.60)
    with caplog.at_level(logging.WARNING, logger='src.analysis.meta_labeler'):
        decision, conf = ml.filter(signal=1.0, features={})
    # Model must have been called (not short-circuited)
    ml.model.predict_proba.assert_called_once()
    # Must still produce a valid decision
    assert decision in ('PASS', 'BLOCK')
    # Warning about missing primary-model features must be logged
    assert any('missing' in r.message.lower() for r in caplog.records), (
        "Must warn when primary-model features are absent"
    )


def test_filter_empty_dict_neutral_priors_correct_values():
    """Verify neutral priors: prob_base=0.5, prob_trend=0.5, regime=0."""
    ml = _make_ml(p_win=0.75)
    captured_X = {}

    def capture_proba(X):
        captured_X['X'] = X.copy()
        return np.array([[0.25, 0.75]])

    ml.model.predict_proba.side_effect = capture_proba
    ml.filter(signal=1.0, features={})
    X = captured_X['X']
    assert float(X.iloc[0]['prob_base']) == pytest.approx(0.5, abs=1e-6)
    assert float(X.iloc[0]['prob_trend']) == pytest.approx(0.5, abs=1e-6)
    assert float(X.iloc[0]['regime']) == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 7. features with all 23 META_FEATURES — model receives correct shape
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_all_23_features_model_receives_correct_column_order():
    """When all META_FEATURES are supplied, model receives X with exactly those 23 cols in order."""
    ml = _make_ml(p_win=0.75)
    ml.filter(signal=1.0, features=_full_features())
    call_args = ml.model.predict_proba.call_args
    X_passed = call_args[0][0]
    assert list(X_passed.columns) == META_FEATURES, (
        "Columns passed to model must exactly match META_FEATURES in order"
    )
    assert X_passed.shape == (1, 23)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Caller-supplied threshold=0.99 — most predictions become BLOCK
#    NOTE: threshold parameter is a FALLBACK only when confidence_threshold is
#    not set (None). When instance threshold is 0.60, caller's 0.99 is ignored.
#    This test verifies that when instance threshold is cleared, caller wins.
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_caller_threshold_99_used_when_no_instance_threshold():
    """When confidence_threshold=None, caller threshold=0.99 is used as the bar."""
    ml = _make_ml(p_win=0.80)
    ml.confidence_threshold = None  # clear instance threshold so caller's is used
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.99)
    # p_win=0.80 < threshold=0.99 => BLOCK
    assert decision == 'BLOCK', (
        f"p_win=0.80 < caller threshold=0.99 must BLOCK; got {decision}"
    )
    assert conf == pytest.approx(0.80, abs=1e-4)


def test_filter_caller_threshold_99_ignored_when_instance_threshold_set():
    """When instance threshold=0.60 is set, caller threshold=0.99 is silently ignored."""
    ml = _make_ml(p_win=0.80, threshold=0.60)
    # Even though caller asks for 0.99, effective threshold is 0.60 (instance wins)
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.99)
    # 0.80 >= 0.60 -> PASS (instance threshold wins over caller's 0.99)
    assert decision == 'PASS', (
        "Instance threshold=0.60 must win over caller threshold=0.99; got BLOCK"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Caller-supplied threshold=0.01 — most predictions become PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_caller_threshold_01_used_when_no_instance_threshold():
    """When confidence_threshold=None, caller threshold=0.01 allows low-confidence trades."""
    ml = _make_ml(p_win=0.15)
    ml.confidence_threshold = None
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.01)
    # p_win=0.15 >= threshold=0.01 => PASS
    assert decision == 'PASS', (
        f"p_win=0.15 >= caller threshold=0.01 must PASS; got {decision}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Instance self.confidence_threshold overrides caller threshold parameter
# ─────────────────────────────────────────────────────────────────────────────

def test_instance_threshold_overrides_caller_threshold():
    """self.confidence_threshold=0.80 overrides caller's threshold=0.40."""
    ml = _make_ml(p_win=0.70, threshold=0.80)  # instance bar is 0.80
    # p_win=0.70 < 0.80 (instance) but p_win=0.70 >= 0.40 (caller)
    decision, _ = ml.filter(signal=1.0, features=_full_features(), threshold=0.40)
    assert decision == 'BLOCK', (
        "Instance threshold=0.80 must override caller=0.40; p_win=0.70 must BLOCK"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-REGRESSION 1: instance confidence_threshold=0.0 (falsy) causes fallback
# Line 156: `getattr(self, 'confidence_threshold', None) or threshold`
# 0.0 is falsy — if a model calibrates to 0.0, the fallback 0.60 is used silently.
# This test documents and locks the current behavior as a known limitation.
# ─────────────────────────────────────────────────────────────────────────────

def test_bug_instance_threshold_zero_falsy_falls_back_to_caller(caplog):
    """
    BUG-REGRESSION: instance threshold=0.0 is falsy; 'or threshold' selects
    caller's 0.60 default instead. A calibrated 0.0 threshold is silently overridden.
    This test documents the behavior so any future fix is caught.
    """
    ml = _make_ml(p_win=0.05, threshold=0.0)
    # With instance threshold=0.0 (intended to mean "always PASS"),
    # the 'or threshold' fallback applies caller's default 0.60
    # so p_win=0.05 < 0.60 -> BLOCK (not PASS as the zero threshold would imply)
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    # Document current (buggy) behavior: BLOCK because 0.0 is falsy
    assert decision == 'BLOCK', (
        "BUG: instance threshold=0.0 (falsy) causes fallback to caller 0.60; "
        "p_win=0.05 < 0.60 -> BLOCK. Fix: use `is None` check instead of `or`."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Model returns proba with shape (1, 1) — one-class model
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_one_class_model_proba_shape_1_1():
    """Model returning shape (1,1) probability: code uses proba[0] as p_win."""
    ml = _make_ml(p_win=0.99)  # will be overridden below
    # Override: one-class model, single probability column
    ml.model.predict_proba.return_value = np.array([[0.82]])  # shape (1,1)
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    # len(proba[0]) == 1 so p_win = proba[0][0] = 0.82 >= 0.60 -> PASS
    assert decision == 'PASS', f"One-class model p_win=0.82 >= 0.60 must PASS; got {decision}"
    assert conf == pytest.approx(0.82, abs=1e-4)


def test_filter_one_class_model_proba_blocks_below_threshold():
    """One-class model with low probability must BLOCK."""
    ml = _make_ml(p_win=0.99)
    ml.confidence_threshold = None
    ml.model.predict_proba.return_value = np.array([[0.40]])  # single class, low prob
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    assert decision == 'BLOCK', f"One-class p_win=0.40 < 0.60 must BLOCK; got {decision}"
    assert conf == pytest.approx(0.40, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Model returns proba with shape (1, 2) — index 1 = win probability
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_two_class_model_uses_index_1_as_p_win():
    """Standard two-class model: proba[:, 1] is the win probability (class 1)."""
    ml = _make_ml(p_win=0.99)
    ml.confidence_threshold = None
    # Explicitly set shape (1,2): [p_lose, p_win]
    ml.model.predict_proba.return_value = np.array([[0.25, 0.75]])
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    assert conf == pytest.approx(0.75, abs=1e-4), "Must extract index-1 from two-class proba"
    assert decision == 'PASS'


def test_filter_two_class_model_high_lose_prob_blocks():
    """Two-class model with high p_lose (index 0), low p_win (index 1) must BLOCK."""
    ml = _make_ml(p_win=0.99)
    ml.confidence_threshold = None
    ml.model.predict_proba.return_value = np.array([[0.85, 0.15]])
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    assert decision == 'BLOCK'
    assert conf == pytest.approx(0.15, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Model returns NaN probability — must fail CLOSED (BLOCK)
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_nan_probability_blocks_fail_closed():
    """NaN in model output: nan >= threshold is False -> BLOCK (correct fail-CLOSED)."""
    ml = _make_ml(p_win=0.99)
    ml.confidence_threshold = None
    ml.model.predict_proba.return_value = np.array([[0.3, float('nan')]])
    decision, conf = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    assert decision == 'BLOCK', "NaN probability must produce BLOCK (fail-CLOSED)"
    import math
    assert math.isnan(conf) or conf == 0.0, "Confidence should be NaN or 0.0 on NaN proba"


def test_filter_all_nan_probability_blocks():
    """Both classes NaN: must BLOCK regardless."""
    ml = _make_ml(p_win=0.99)
    ml.confidence_threshold = None
    ml.model.predict_proba.return_value = np.array([[float('nan'), float('nan')]])
    decision, _ = ml.filter(signal=1.0, features=_full_features(), threshold=0.60)
    assert decision == 'BLOCK', "All-NaN proba must BLOCK"


# ─────────────────────────────────────────────────────────────────────────────
# 14. batch_filter with empty signals Series — returns empty result, no error
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_filter_empty_signals_returns_empty_dataframe():
    """Empty signals Series + empty features_df: must return empty DataFrame, no exception."""
    ml = _make_ml(p_win=0.75)
    ml.model.predict_proba.return_value = np.empty((0, 2))
    signals = pd.Series([], dtype=float)
    feats_df = pd.DataFrame(columns=META_FEATURES)
    result = ml.batch_filter(signals, feats_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0, f"Expected empty result, got {len(result)} rows"
    assert 'decision' in result.columns
    assert 'confidence' in result.columns
    assert 'filtered_signal' in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# 15. batch_filter with signals.index != features_df.index — index alignment
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_filter_mismatched_index_uses_positional_alignment():
    """
    When signals.index != features_df.index, batch_filter uses .values (positional)
    for signals and features_df.index for the output. No crash; output inherits
    features_df.index (not signals.index).
    """
    ml = _make_ml(p_win=0.75, threshold=0.60)
    n = 3
    ml.model.predict_proba.return_value = np.array([[0.25, 0.75]] * n)

    signals = pd.Series([1.0, -1.0, 1.0], index=[100, 200, 300])  # index 100/200/300
    feats = pd.DataFrame({f: [0.0] * n for f in META_FEATURES}, index=[0, 1, 2])  # index 0/1/2

    result = ml.batch_filter(signals, feats)

    # Output index must be features_df.index (0,1,2), not signals.index (100,200,300)
    assert list(result.index) == [0, 1, 2], (
        f"Output index must equal features_df.index; got {list(result.index)}"
    )
    assert len(result) == 3
    # All p_win=0.75 >= 0.60 -> PASS
    assert (result['decision'] == 'PASS').all()


# ─────────────────────────────────────────────────────────────────────────────
# 16. _load() with valid model but missing meta JSON — uses default threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_load_missing_meta_json_uses_default_threshold(tmp_path):
    """Valid model file + no _meta.json: confidence_threshold must equal CONFIDENCE_THRESHOLD."""
    import joblib
    from sklearn.dummy import DummyClassifier

    # Create a minimal valid model file
    model_path = tmp_path / 'meta_labeler.joblib'
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit([[0] * 23], [1])
    joblib.dump(dummy_clf, str(model_path))

    # No _meta.json exists alongside the model
    meta_json = tmp_path / 'meta_labeler_meta.json'
    assert not meta_json.exists()

    with patch('src.utils.model_integrity.verify_and_load_bytes',
               side_effect=lambda p: open(p, 'rb').read()):
        ml = MetaLabeler(model_path=str(model_path))

    assert ml.is_loaded is True
    assert ml.confidence_threshold == pytest.approx(CONFIDENCE_THRESHOLD, abs=1e-6), (
        f"Missing meta JSON must fall back to default {CONFIDENCE_THRESHOLD}; "
        f"got {ml.confidence_threshold}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 17. _load() with valid model + meta JSON containing optimal_threshold=0.65
# ─────────────────────────────────────────────────────────────────────────────

def test_load_meta_json_optimal_threshold_applied(tmp_path):
    """Meta JSON with optimal_threshold=0.65: instance threshold must be 0.65."""
    import joblib
    from sklearn.dummy import DummyClassifier

    model_path = tmp_path / 'meta_labeler.joblib'
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit([[0] * 23], [1])
    joblib.dump(dummy_clf, str(model_path))

    meta_json = tmp_path / 'meta_labeler_meta.json'
    meta_json.write_text(json.dumps({'optimal_threshold': 0.65, 'accuracy': 60.0}),
                         encoding='utf-8')

    with patch('src.utils.model_integrity.verify_and_load_bytes',
               side_effect=lambda p: open(p, 'rb').read()):
        ml = MetaLabeler(model_path=str(model_path))

    assert ml.confidence_threshold == pytest.approx(0.65, abs=1e-6), (
        f"Meta JSON optimal_threshold=0.65 must set instance threshold; "
        f"got {ml.confidence_threshold}"
    )


def test_load_meta_json_confidence_threshold_key_also_accepted(tmp_path):
    """Meta JSON with 'confidence_threshold' key (fallback key) must also be accepted."""
    import joblib
    from sklearn.dummy import DummyClassifier

    model_path = tmp_path / 'meta_labeler.joblib'
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit([[0] * 23], [1])
    joblib.dump(dummy_clf, str(model_path))

    meta_json = tmp_path / 'meta_labeler_meta.json'
    meta_json.write_text(json.dumps({'confidence_threshold': 0.55}), encoding='utf-8')

    with patch('src.utils.model_integrity.verify_and_load_bytes',
               side_effect=lambda p: open(p, 'rb').read()):
        ml = MetaLabeler(model_path=str(model_path))

    assert ml.confidence_threshold == pytest.approx(0.55, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 18. _load() with corrupted meta JSON — falls back to default, logs warning
# ─────────────────────────────────────────────────────────────────────────────

def test_load_corrupted_meta_json_falls_back_to_default(tmp_path, caplog):
    """Corrupted (invalid JSON) meta file: must log WARNING and keep default threshold."""
    import joblib
    from sklearn.dummy import DummyClassifier

    model_path = tmp_path / 'meta_labeler.joblib'
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit([[0] * 23], [1])
    joblib.dump(dummy_clf, str(model_path))

    meta_json = tmp_path / 'meta_labeler_meta.json'
    meta_json.write_text('NOT VALID JSON {{{', encoding='utf-8')

    with caplog.at_level(logging.WARNING, logger='src.analysis.meta_labeler'):
        with patch('src.utils.model_integrity.verify_and_load_bytes',
                   side_effect=lambda p: open(p, 'rb').read()):
            ml = MetaLabeler(model_path=str(model_path))

    # Must still load the model
    assert ml.is_loaded is True
    # Threshold must be unchanged (default)
    assert ml.confidence_threshold == pytest.approx(CONFIDENCE_THRESHOLD, abs=1e-6), (
        "Corrupted meta JSON must not change threshold from default"
    )
    # Must have logged a warning
    assert any('warning' in r.levelname.lower() or r.levelno >= logging.WARNING
               for r in caplog.records), (
        "Corrupted meta JSON must emit a WARNING log entry"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional safety invariant: batch_filter must NEVER fail-OPEN
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_filter_fails_closed_when_model_not_loaded():
    """batch_filter with no model loaded must return all-BLOCK DataFrame."""
    ml = MetaLabeler.__new__(MetaLabeler)
    ml.model_path = '/nonexistent'
    ml.model = None
    ml.is_loaded = False
    ml.confidence_threshold = CONFIDENCE_THRESHOLD

    signals = pd.Series([1.0, 1.0, -1.0])
    feats_df = pd.DataFrame({f: [0.0] * 3 for f in META_FEATURES})
    result = ml.batch_filter(signals, feats_df)

    assert (result['decision'] == 'BLOCK').all(), "Unloaded model must BLOCK all in batch"
    assert (result['filtered_signal'] == 0.0).all()
    assert (result['confidence'] == 0.0).all()


def test_batch_filter_nan_proba_blocks_affected_rows():
    """batch_filter where model returns NaN for some rows: those rows must BLOCK."""
    ml = _make_ml(p_win=0.75)
    ml.confidence_threshold = None
    n = 4
    proba = np.array([[0.25, 0.75], [0.30, float('nan')], [0.20, 0.80], [0.40, float('nan')]])
    ml.model.predict_proba.return_value = proba

    signals = pd.Series([1.0, 1.0, 1.0, 1.0])
    feats_df = pd.DataFrame({f: [0.0] * n for f in META_FEATURES})
    result = ml.batch_filter(signals, feats_df, threshold=0.60)

    # Rows 0 and 2: p_win=0.75/0.80 >= 0.60 -> PASS; filtered_signal preserved
    # Rows 1 and 3: p_win=NaN -> nan >= 0.60 is False -> BLOCK; filtered_signal=0.0
    assert result.iloc[0]['decision'] == 'PASS'
    assert result.iloc[0]['filtered_signal'] == pytest.approx(1.0)
    assert result.iloc[1]['decision'] == 'BLOCK'
    assert result.iloc[1]['filtered_signal'] == pytest.approx(0.0)
    assert result.iloc[2]['decision'] == 'PASS'
    assert result.iloc[3]['decision'] == 'BLOCK'


def test_batch_filter_feature_count_mismatch_blocks_all():
    """batch_filter with model expecting different feature count must BLOCK everything."""
    ml = _make_ml(n_features_in=10)  # model expects 10 but META_FEATURES has 23

    signals = pd.Series([1.0, 1.0, -1.0])
    feats_df = pd.DataFrame({f: [0.0] * 3 for f in META_FEATURES})
    result = ml.batch_filter(signals, feats_df)

    assert (result['decision'] == 'BLOCK').all(), (
        "Feature count mismatch must BLOCK all signals in batch"
    )
    assert (result['filtered_signal'] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# BUG-REGRESSION 2: filter() on exception must return 0.0 confidence, not raise
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_exception_during_inference_returns_block_zero_confidence():
    """Inference exception: must return ('BLOCK', 0.0), never propagate."""
    ml = _make_ml(p_win=0.99)
    ml.model.predict_proba.side_effect = MemoryError("OOM")
    decision, conf = ml.filter(signal=1.0, features=_full_features())
    assert decision == 'BLOCK'
    assert conf == 0.0
    # Must not re-raise — calling code must receive the tuple, not an exception


# ─────────────────────────────────────────────────────────────────────────────
# BUG-REGRESSION 3: batch_filter error sentinel value is -1.0 (not 0.0)
# The batch error path sets confidence=-1.0 so callers can detect error state
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_filter_exception_sentinel_confidence_is_negative_one():
    """
    On batch_filter exception, confidence sentinel is -1.0 (not 0.0).
    Callers can detect the error state via confidence < 0.
    """
    ml = _make_ml(p_win=0.99)
    ml.model.predict_proba.side_effect = RuntimeError("batch inference failure")

    signals = pd.Series([1.0, -1.0, 1.0])
    feats_df = pd.DataFrame({f: [0.0] * 3 for f in META_FEATURES})
    result = ml.batch_filter(signals, feats_df)

    assert (result['decision'] == 'BLOCK').all()
    assert (result['filtered_signal'] == 0.0).all()
    # Sentinel: confidence=-1.0 flags error state
    assert (result['confidence'] == -1.0).all(), (
        "batch_filter exception path must use confidence=-1.0 as error sentinel"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-REGRESSION 4: list features with no dicts — falls back to empty dict
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_list_of_non_dicts_falls_back_gracefully():
    """
    list features where elements are not dicts (e.g., empty list, list of floats):
    code falls to elif isinstance(features, dict) which is False -> feature_row stays {}.
    Must not raise AttributeError — must still run with neutral priors.
    """
    ml = _make_ml(p_win=0.75)
    # list of floats: last element is not a dict -> branch skipped -> feature_row = {}
    decision, conf = ml.filter(signal=1.0, features=[1.0, 2.0, 3.0])
    # No crash; model still runs (with empty feature_row -> neutral priors)
    assert decision in ('PASS', 'BLOCK'), "Non-dict list must not raise"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full happy path end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_happy_path_pass():
    """Full happy path: all 23 features, high p_win, reasonable threshold -> PASS."""
    ml = _make_ml(p_win=0.75, threshold=0.60)
    decision, conf = ml.filter(signal=1.0, features=_full_features())
    assert decision == 'PASS'
    assert conf == pytest.approx(0.75, abs=1e-4)


def test_filter_happy_path_block():
    """Full happy path: all 23 features, low p_win, standard threshold -> BLOCK."""
    ml = _make_ml(p_win=0.45, threshold=0.60)
    decision, conf = ml.filter(signal=1.0, features=_full_features())
    assert decision == 'BLOCK'
    assert conf == pytest.approx(0.45, abs=1e-4)


def test_batch_filter_happy_path_mixed_decisions():
    """batch_filter: rows above threshold PASS and preserve signal; below BLOCK."""
    ml = _make_ml(p_win=0.0)  # will be overridden
    ml.confidence_threshold = None
    n = 5
    # Alternating high/low p_win
    p_wins = [0.80, 0.30, 0.70, 0.20, 0.90]
    proba = np.array([[1 - p, p] for p in p_wins])
    ml.model.predict_proba.return_value = proba

    signals = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
    feats_df = pd.DataFrame({f: [0.0] * n for f in META_FEATURES})

    result = ml.batch_filter(signals, feats_df, threshold=0.60)

    expected_decisions = ['PASS', 'BLOCK', 'PASS', 'BLOCK', 'PASS']
    assert list(result['decision']) == expected_decisions
    # PASS rows keep original signal; BLOCK rows get 0.0
    assert result.iloc[0]['filtered_signal'] == pytest.approx(1.0)
    assert result.iloc[1]['filtered_signal'] == pytest.approx(0.0)
    assert result.iloc[4]['filtered_signal'] == pytest.approx(1.0)


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
