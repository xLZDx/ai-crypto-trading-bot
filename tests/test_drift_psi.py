"""Phase 6 (2026-05-14) — F2 Concept Drift tests."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.risk import drift_psi as dp  # noqa: E402


def _baseline_for(values: np.ndarray, n_bins: int = 10) -> dict:
    """Build a baseline entry the same way drift_baseline.save_baseline does."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    edges[-1] += 1e-9
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    props = (counts / total).tolist() if total else [0.0] * (len(edges) - 1)
    return {"bin_edges": edges.tolist(), "bin_props": props,
            "mean": float(values.mean()), "std": float(values.std()),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)), "n": int(len(values))}


class _ModeReset(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.pop(dp._MODE_ENV, None)

    def tearDown(self) -> None:
        if self._old is not None:
            os.environ[dp._MODE_ENV] = self._old
        else:
            os.environ.pop(dp._MODE_ENV, None)


class TestPSI(unittest.TestCase):
    def test_identical_distributions_psi_near_zero(self) -> None:
        np.random.seed(0)
        x = np.random.normal(0, 1, 5000)
        b = _baseline_for(x)
        psi = dp.compute_psi(b["bin_props"], b["bin_edges"], x)
        self.assertLess(psi, 0.02, f"identical-sample PSI should be ~0, got {psi}")

    def test_mean_shift_psi_above_warn(self) -> None:
        np.random.seed(0)
        x = np.random.normal(0, 1, 5000)
        b = _baseline_for(x)
        # Shift the actual by 1 sigma — moderate drift
        y = np.random.normal(1.0, 1, 5000)
        psi = dp.compute_psi(b["bin_props"], b["bin_edges"], y)
        self.assertGreater(psi, dp.PSI_WARN_THRESHOLD,
                           f"mean-shifted sample PSI should exceed warn, got {psi}")

    def test_large_shift_psi_above_pause(self) -> None:
        np.random.seed(0)
        x = np.random.normal(0, 1, 5000)
        b = _baseline_for(x)
        # 3-sigma shift — significant drift
        y = np.random.normal(3.0, 1, 5000)
        psi = dp.compute_psi(b["bin_props"], b["bin_edges"], y)
        self.assertGreater(psi, dp.PSI_PAUSE_THRESHOLD,
                           f"large-shift PSI should exceed pause, got {psi}")

    def test_empty_input_returns_zero(self) -> None:
        psi = dp.compute_psi([0.5, 0.5], [0.0, 0.5, 1.0], np.array([]))
        self.assertEqual(psi, 0.0)


class TestWasserstein(unittest.TestCase):
    def test_identical_dist_wd_near_zero(self) -> None:
        np.random.seed(1)
        x = np.random.normal(0, 1, 3000)
        b = _baseline_for(x)
        wd = dp.compute_wasserstein_relative(b["bin_edges"], b["bin_props"], x)
        self.assertLess(wd, 0.1, f"identical-sample WD_rel should be ~0, got {wd}")

    def test_mean_shift_wd_increases(self) -> None:
        np.random.seed(1)
        x = np.random.normal(0, 1, 3000)
        b = _baseline_for(x)
        y_identical = np.random.normal(0, 1, 3000)
        y_shifted = np.random.normal(2.0, 1, 3000)
        wd_id = dp.compute_wasserstein_relative(b["bin_edges"], b["bin_props"], y_identical)
        wd_sh = dp.compute_wasserstein_relative(b["bin_edges"], b["bin_props"], y_shifted)
        self.assertGreater(wd_sh, wd_id,
                           "shifted-sample WD should exceed identical-sample WD")


class TestCheckDrift(_ModeReset):
    def test_no_drift_returns_ok_findings(self) -> None:
        np.random.seed(2)
        baseline_df = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        actual_df   = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        baseline = {"ofi_z": _baseline_for(baseline_df["ofi_z"].values)}
        rep = dp.check_drift(baseline, actual_df)
        self.assertEqual(rep.mode, "warn")
        self.assertEqual(len(rep.findings), 1)
        # Random samples may be ok or warn — just assert it's not pause
        self.assertIn(rep.findings[0].severity, ("ok", "warn"))

    def test_severe_drift_on_hard_feature_pauses_in_enforce(self) -> None:
        os.environ[dp._MODE_ENV] = "enforce"
        np.random.seed(3)
        baseline_df = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        actual_df   = pd.DataFrame({"ofi_z": np.random.normal(5.0, 0.5, 3000)})
        baseline = {"ofi_z": _baseline_for(baseline_df["ofi_z"].values)}
        with self.assertRaises(dp.DriftPauseError):
            dp.check_drift(baseline, actual_df)

    def test_severe_drift_in_warn_mode_does_not_raise(self) -> None:
        os.environ[dp._MODE_ENV] = "warn"
        np.random.seed(4)
        baseline_df = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        actual_df   = pd.DataFrame({"ofi_z": np.random.normal(5.0, 0.5, 3000)})
        baseline = {"ofi_z": _baseline_for(baseline_df["ofi_z"].values)}
        rep = dp.check_drift(baseline, actual_df)
        # Warn mode logs but doesn't raise
        self.assertFalse(rep.pause_triggered)
        self.assertGreater(len(rep.by_severity("pause")), 0)

    def test_off_mode_skips_check(self) -> None:
        os.environ[dp._MODE_ENV] = "off"
        np.random.seed(5)
        actual_df = pd.DataFrame({"ofi_z": np.random.normal(10.0, 1, 3000)})
        baseline = {"ofi_z": _baseline_for(np.random.normal(0, 1, 3000))}
        rep = dp.check_drift(baseline, actual_df)
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.mode, "off")

    def test_non_hard_feature_only_warns_not_pauses(self) -> None:
        """Drift on a raw OHLCV feature (e.g. 'volume') is regime change,
        not model breakage. In enforce mode, only HARD features pause."""
        os.environ[dp._MODE_ENV] = "enforce"
        np.random.seed(6)
        baseline_df = pd.DataFrame({"volume": np.random.normal(1000, 100, 3000)})
        actual_df   = pd.DataFrame({"volume": np.random.normal(5000, 200, 3000)})
        baseline = {"volume": _baseline_for(baseline_df["volume"].values)}
        # volume not in DRIFT_HARD_FEATURES → max severity is "warn", no raise
        rep = dp.check_drift(baseline, actual_df)
        self.assertFalse(rep.pause_triggered)
        self.assertEqual(len(rep.by_severity("pause")), 0)
        self.assertGreater(len(rep.by_severity("warn")), 0)

    def test_hard_features_extra_env_picks_up_new_features(self) -> None:
        """Operator can add features to the hard-feature set at runtime."""
        os.environ[dp._MODE_ENV] = "enforce"
        os.environ["DRIFT_HARD_FEATURES_EXTRA"] = "volume,trades_count"
        np.random.seed(7)
        baseline_df = pd.DataFrame({"volume": np.random.normal(1000, 100, 3000)})
        actual_df   = pd.DataFrame({"volume": np.random.normal(5000, 200, 3000)})
        baseline = {"volume": _baseline_for(baseline_df["volume"].values)}
        try:
            with self.assertRaises(dp.DriftPauseError):
                dp.check_drift(baseline, actual_df)
        finally:
            os.environ.pop("DRIFT_HARD_FEATURES_EXTRA", None)

    def test_report_serializes_to_dict(self) -> None:
        os.environ[dp._MODE_ENV] = "warn"
        import json
        np.random.seed(8)
        baseline_df = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        actual_df   = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 3000)})
        baseline = {"ofi_z": _baseline_for(baseline_df["ofi_z"].values)}
        rep = dp.check_drift(baseline, actual_df)
        d = rep.to_dict()
        json.dumps(d)  # must be JSON-serializable for state file persistence
        self.assertEqual(d["mode"], "warn")
        self.assertIn("findings", d)


class TestBaselineWithBins(unittest.TestCase):
    """Verify the extended drift_baseline.save_baseline still works + adds bins."""

    def test_save_baseline_now_persists_bin_edges_and_props(self) -> None:
        from src.risk import drift_baseline as db
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            db.BASELINES_DIR = type(db.BASELINES_DIR)(td)
            df = pd.DataFrame({
                "ofi_z": np.random.normal(0, 1, 500),
                "funding_z": np.random.normal(0, 0.5, 500),
            })
            payload = db.save_baseline("trend", "1h", df)
            self.assertIn("features", payload)
            for f in ("ofi_z", "funding_z"):
                entry = payload["features"][f]
                self.assertIn("bin_edges", entry)
                self.assertIn("bin_props", entry)
                self.assertGreater(len(entry["bin_edges"]), 1)
                self.assertEqual(len(entry["bin_props"]),
                                 len(entry["bin_edges"]) - 1)


if __name__ == "__main__":
    unittest.main()
