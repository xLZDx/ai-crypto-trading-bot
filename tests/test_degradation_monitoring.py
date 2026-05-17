"""
Unit tests for degradation-monitoring features — P2, P3+P4, P5.

P2  — Two-tier drift enforcement (drift_psi._enforce_features,
       drift_monitor.CellState.consecutive_pause_count,
       drift_monitor.is_drift_paused two-tier logic)

P5  — Per-strategy regression guard (_per_strategy_regressions)

P3+P4 — KPI gate overfit_ratio + wf_fold_scores slope gate
         (_check_thresholds)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── P2: drift_psi._enforce_features ─────────────────────────────────────────

class TestEnforceFeatures:
    def test_empty_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DRIFT_ENFORCE_FEATURES", raising=False)
        from src.risk.drift_psi import _enforce_features
        assert _enforce_features() == frozenset()

    def test_returns_intersection_with_hard_features(self, monkeypatch):
        monkeypatch.setenv("DRIFT_ENFORCE_FEATURES", "ofi_z,funding_z,not_a_real_feature")
        from src.risk.drift_psi import _enforce_features, DRIFT_HARD_FEATURES
        result = _enforce_features()
        assert "ofi_z" in result
        assert "funding_z" in result
        assert "not_a_real_feature" not in result
        assert result <= DRIFT_HARD_FEATURES

    def test_empty_string_returns_empty(self, monkeypatch):
        monkeypatch.setenv("DRIFT_ENFORCE_FEATURES", "   ")
        from src.risk.drift_psi import _enforce_features
        assert _enforce_features() == frozenset()


class TestDriftFindingEnforceFlag:
    def _make_baseline(self, feature: str, n_bins: int = 10) -> dict:
        import numpy as np
        vals = np.linspace(0.0, 1.0, n_bins + 1)
        edges = vals.tolist()
        props = [1.0 / n_bins] * n_bins
        return {feature: {"bin_edges": edges, "bin_props": props}}

    def _make_actual_df(self, feature: str, n: int = 100):
        import pandas as pd, numpy as np
        return pd.DataFrame({feature: np.random.uniform(0, 1, n)})

    def test_is_enforce_feature_set_for_enforce_tier(self, monkeypatch):
        monkeypatch.setenv("DRIFT_ENFORCE_FEATURES", "ofi_z")
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "warn")
        import pandas as pd, numpy as np
        from src.risk.drift_psi import check_drift

        baseline = self._make_baseline("ofi_z")
        actual = pd.DataFrame({"ofi_z": np.random.uniform(0, 1, 100)})
        rep = check_drift(baseline, actual, force_mode="warn")

        findings = {f.feature: f for f in rep.findings}
        assert "ofi_z" in findings
        assert findings["ofi_z"].is_enforce_feature is True

    def test_is_enforce_feature_false_for_non_enforce_tier(self, monkeypatch):
        monkeypatch.setenv("DRIFT_ENFORCE_FEATURES", "ofi_z")
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "warn")
        import pandas as pd, numpy as np
        from src.risk.drift_psi import check_drift

        baseline = self._make_baseline("atr_14")
        actual = pd.DataFrame({"atr_14": np.random.uniform(0, 1, 100)})
        rep = check_drift(baseline, actual, force_mode="warn")

        findings = {f.feature: f for f in rep.findings}
        if "atr_14" in findings:
            assert findings["atr_14"].is_enforce_feature is False


# ── P2: drift_monitor.is_drift_paused — two-tier logic ──────────────────────

class TestIsDriftPausedTwoTier:
    """Test is_drift_paused() with mocked cached state."""

    def _cell(self, model, tf, pause_count, consecutive, enforce_feats=None):
        findings = []
        if pause_count > 0:
            feat = enforce_feats[0] if enforce_feats else "atr_14"
            findings.append({
                "feature": feat,
                "severity": "pause",
                "is_hard": True,
                "is_enforce": bool(enforce_feats),
                "psi": 0.30,
                "wasserstein_rel": 0.10,
                "note": "",
            })
        return {
            "model": model,
            "tf": tf,
            "report": {"pause_count": pause_count, "findings": findings},
            "consecutive_pause_count": consecutive,
        }

    def test_not_enforcing_returns_false(self, monkeypatch):
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "warn")
        from src.risk.drift_monitor import is_drift_paused
        paused, reason = is_drift_paused("base", "1h")
        assert paused is False
        assert "not enforcing" in reason

    def test_enforce_tier_halts_immediately_on_first_poll(self, monkeypatch):
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "enforce")
        monkeypatch.setenv("DRIFT_ENFORCE_FEATURES", "ofi_z")
        state = {
            "cells": [self._cell("base", "1h", pause_count=1,
                                  consecutive=1, enforce_feats=["ofi_z"])]
        }
        from src.risk.drift_monitor import is_drift_paused
        with patch("src.risk.drift_monitor.get_cached_state", return_value=state):
            paused, reason = is_drift_paused("base", "1h")
        assert paused is True
        assert "enforce-tier" in reason

    def test_confirm_tier_needs_3_consecutive(self, monkeypatch):
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "enforce")
        monkeypatch.delenv("DRIFT_ENFORCE_FEATURES", raising=False)
        from src.risk.drift_monitor import is_drift_paused

        for consecutive, expected in [(1, False), (2, False), (3, True), (4, True)]:
            state = {"cells": [self._cell("trend", "4h", pause_count=1,
                                           consecutive=consecutive)]}
            with patch("src.risk.drift_monitor.get_cached_state", return_value=state):
                paused, _ = is_drift_paused("trend", "4h")
            assert paused is expected, f"consecutive={consecutive}: expected {expected}"

    def test_clean_cell_returns_false(self, monkeypatch):
        monkeypatch.setenv("LLM_DRIFT_PAUSE", "enforce")
        state = {"cells": [self._cell("base", "1h", pause_count=0, consecutive=0)]}
        from src.risk.drift_monitor import is_drift_paused
        with patch("src.risk.drift_monitor.get_cached_state", return_value=state):
            paused, reason = is_drift_paused("base", "1h")
        assert paused is False
        assert "cell_clean" in reason


# ── P5: _per_strategy_regressions ───────────────────────────────────────────

class TestPerStrategyRegressions:
    def test_one_regresses_one_improves(self):
        from src.engine.auto_retrain import _per_strategy_regressions
        before = {"RSI_MR": 0.50, "Trend_ML": 0.60}
        after  = {"RSI_MR": 0.42, "Trend_ML": 0.70}
        # RSI_MR: 0.42 < 0.50 * 0.95 = 0.475 → regressed
        # Trend_ML improved
        result = _per_strategy_regressions(before, after, tolerance=0.05)
        assert "RSI_MR" in result
        assert "Trend_ML" not in result

    def test_new_strategy_excluded(self):
        from src.engine.auto_retrain import _per_strategy_regressions
        before = {"RSI_MR": 0.50}
        after  = {"RSI_MR": 0.48, "NewStrat": 0.55}
        # RSI_MR: 0.48 >= 0.50 * 0.95 = 0.475 → OK
        # NewStrat: not in before → excluded
        result = _per_strategy_regressions(before, after, tolerance=0.05)
        assert "NewStrat" not in result
        assert "RSI_MR" not in result

    def test_all_regress(self):
        from src.engine.auto_retrain import _per_strategy_regressions
        before = {"A": 0.60, "B": 0.55}
        after  = {"A": 0.40, "B": 0.35}
        result = _per_strategy_regressions(before, after, tolerance=0.05)
        assert set(result) == {"A", "B"}

    def test_empty_before_returns_empty(self):
        from src.engine.auto_retrain import _per_strategy_regressions
        result = _per_strategy_regressions({}, {"A": 0.60}, tolerance=0.05)
        assert result == []

    def test_regression_verdict_overrides_positive_avg(self):
        """If one strategy drops individually, verdict must be 'regression'
        even when the system-wide average improves."""
        from src.engine.auto_retrain import _per_strategy_regressions
        before = {"A": 1.00, "B": 0.10}
        after  = {"A": 0.50, "B": 2.00}
        # System avg: before=0.55, after=1.25 (improved!)
        # But A dropped from 1.00 to 0.50 (50% below old*0.95=0.95) → regression
        result = _per_strategy_regressions(before, after, tolerance=0.05)
        assert "A" in result


# ── P3+P4: kpi_gate._check_thresholds ───────────────────────────────────────

class TestCheckThresholds:
    def _make_run(self, **kwargs):
        from src.engine.kpi_gate import TrainingResult
        defaults = dict(
            model_key="base", tf="1h",
            started_at=0.0, finished_at=1.0,
            artifact_path="dummy.joblib",
        )
        defaults.update(kwargs)
        return TrainingResult(**defaults)

    def test_overfit_ratio_below_threshold_passes(self):
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(overfit_ratio=0.15)
        missed = _check_thresholds(run, {"overfit_ratio": 0.20})
        assert not missed

    def test_overfit_ratio_above_threshold_fails(self):
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(overfit_ratio=0.25)
        missed = _check_thresholds(run, {"overfit_ratio": 0.20})
        assert any("overfit_ratio" in m for m in missed)

    def test_overfit_ratio_missing_counts_as_missed(self):
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(overfit_ratio=None)
        missed = _check_thresholds(run, {"overfit_ratio": 0.20})
        assert any("overfit_ratio" in m for m in missed)

    def test_negative_fold_slope_flagged(self):
        from src.engine.kpi_gate import _check_thresholds
        # Scores clearly declining: last < first and relative slope << -0.02
        run = self._make_run(wf_fold_scores=[0.58, 0.54, 0.50, 0.46])
        missed = _check_thresholds(run, {})
        assert any("wf_fold_slope:negative" in m for m in missed)

    def test_noisy_slope_with_last_greater_than_first_not_flagged(self):
        from src.engine.kpi_gate import _check_thresholds
        # Last fold (0.56) > first fold (0.52) → sanity gate prevents false positive
        run = self._make_run(wf_fold_scores=[0.52, 0.48, 0.50, 0.56])
        missed = _check_thresholds(run, {})
        assert not any("wf_fold_slope" in m for m in missed)

    def test_fewer_than_3_folds_no_slope_check(self):
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(wf_fold_scores=[0.58, 0.40])
        missed = _check_thresholds(run, {})
        assert not any("wf_fold_slope" in m for m in missed)

    def test_stable_folds_not_flagged(self):
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(wf_fold_scores=[0.55, 0.54, 0.55, 0.56, 0.55])
        missed = _check_thresholds(run, {})
        assert not any("wf_fold_slope" in m for m in missed)

    def test_wf_acc_min_threshold_still_works(self):
        """Existing MIN thresholds unaffected by P3+P4 additions."""
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(wf_acc=0.48)
        missed = _check_thresholds(run, {"wf_acc": 50.0})
        assert any("wf_acc" in m for m in missed)

    def test_three_strike_scenario_with_overfit_ratio(self):
        """A run failing overfit_ratio contributes to the 3-strike retirement."""
        from src.engine.kpi_gate import _check_thresholds
        run = self._make_run(overfit_ratio=0.30, wf_acc=0.55)
        thresholds = {"wf_acc": 50.0, "overfit_ratio": 0.20}
        missed = _check_thresholds(run, thresholds)
        assert any("overfit_ratio" in m for m in missed)
        assert not any("wf_acc" in m for m in missed)
