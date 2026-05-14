"""Phase 6b (2026-05-14) — drift_monitor regression tests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.risk import drift_monitor as dm  # noqa: E402


def _baseline_payload(values: np.ndarray, feat: str = "ofi_z") -> dict:
    """Build a baseline JSON the same way drift_baseline.save_baseline does
    (mean/std/q05/q95/n + bin_edges + bin_props for PSI)."""
    quantiles = np.linspace(0, 1, 11)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    edges[-1] += 1e-9
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    props = (counts / total).tolist() if total else [0.0] * (len(edges) - 1)
    return {
        "model_key": "test",
        "timeframe": "1h",
        "saved_at": "2026-05-14T00:00:00+00:00",
        "feature_count": 1,
        "features": {
            feat: {
                "mean": float(values.mean()), "std": float(values.std()),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "n": int(len(values)),
                "bin_edges": edges.tolist(),
                "bin_props": props,
            }
        }
    }


class _TmpRoot(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.tmp.name)
        self.baselines = self.tmp_root / "baselines"
        self.training_runs = self.tmp_root / "training_runs"
        self.state_file = self.tmp_root / "drift_state.json"
        self.baselines.mkdir(parents=True, exist_ok=True)
        self.training_runs.mkdir(parents=True, exist_ok=True)
        self._patches = [
            mock.patch.object(dm, "BASELINES_DIR", self.baselines),
            mock.patch.object(dm, "TRAINING_RUNS_DIR", self.training_runs),
            mock.patch.object(dm, "STATE_FILE", self.state_file),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()


class TestRunOnceShape(_TmpRoot):
    def test_no_baselines_yields_empty_state(self) -> None:
        state = dm.run_once()
        self.assertEqual(state["cell_count"], 0)
        self.assertEqual(state["cells"], [])
        # Must still persist a valid state file
        self.assertTrue(self.state_file.exists())

    def test_baseline_without_actual_yields_no_actual_note(self) -> None:
        np.random.seed(0)
        payload = _baseline_payload(np.random.normal(0, 1, 1000))
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        state = dm.run_once()
        self.assertEqual(state["cell_count"], 1)
        cell = state["cells"][0]
        self.assertEqual(cell["model"], "trend")
        self.assertEqual(cell["tf"], "1h")
        self.assertEqual(cell["actual_source"], "no_actual")

    def test_baseline_with_matching_parquet_runs_drift_check(self) -> None:
        np.random.seed(1)
        ref = np.random.normal(0, 1, 2000)
        payload = _baseline_payload(ref, feat="ofi_z")
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        # Identical-distribution actuals → no drift
        actual_df = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 2000)})
        actual_df.to_parquet(self.training_runs / "trend__1h.parquet")
        state = dm.run_once()
        cell = state["cells"][0]
        self.assertEqual(cell["actual_source"], "training_runs")
        rep = cell["report"]
        self.assertEqual(rep.get("pause_count"), 0)

    def test_baseline_with_shifted_actual_flags_pause(self) -> None:
        """ofi_z is a HARD feature; severe shift → severity=pause."""
        np.random.seed(2)
        ref = np.random.normal(0, 1, 2000)
        payload = _baseline_payload(ref, feat="ofi_z")
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        # 5σ shift on a hard feature → pause
        actual_df = pd.DataFrame({"ofi_z": np.random.normal(5.0, 0.5, 2000)})
        actual_df.to_parquet(self.training_runs / "trend__1h.parquet")
        state = dm.run_once()
        cell = state["cells"][0]
        rep = cell["report"]
        self.assertGreaterEqual(rep.get("pause_count", 0), 1,
                                f"expected pause on severe drift, got {rep}")


class TestCachedRead(_TmpRoot):
    def test_get_cached_state_returns_saved_state(self) -> None:
        np.random.seed(3)
        payload = _baseline_payload(np.random.normal(0, 1, 1000))
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        state = dm.run_once()
        cached = dm.get_cached_state()
        self.assertEqual(cached["cell_count"], state["cell_count"])
        self.assertEqual(len(cached["cells"]), len(state["cells"]))

    def test_get_cached_state_returns_empty_when_no_file(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
        cached = dm.get_cached_state()
        self.assertEqual(cached, {})


class TestParseBaselineFilename(unittest.TestCase):
    def test_valid_stem(self) -> None:
        self.assertEqual(dm._parse_baseline_filename("trend__1h"), ("trend", "1h"))

    def test_invalid_stem(self) -> None:
        self.assertIsNone(dm._parse_baseline_filename("not_a_pattern"))


class TestIsDriftPaused(_TmpRoot):
    """Phase 6c — bot-facing helper. Read-only consumer of drift_state.json."""

    def setUp(self) -> None:
        super().setUp()
        self._old_mode = os.environ.pop("LLM_DRIFT_PAUSE", None)

    def tearDown(self) -> None:
        if self._old_mode is not None:
            os.environ["LLM_DRIFT_PAUSE"] = self._old_mode
        else:
            os.environ.pop("LLM_DRIFT_PAUSE", None)
        super().tearDown()

    def test_warn_mode_never_pauses(self) -> None:
        os.environ["LLM_DRIFT_PAUSE"] = "warn"
        paused, why = dm.is_drift_paused("trend", "1h")
        self.assertFalse(paused)
        self.assertIn("warn", why)

    def test_off_mode_never_pauses(self) -> None:
        os.environ["LLM_DRIFT_PAUSE"] = "off"
        paused, why = dm.is_drift_paused("trend", "1h")
        self.assertFalse(paused)

    def test_enforce_no_baselines_does_not_pause(self) -> None:
        """If nothing's been trained, there's nothing to drift FROM."""
        os.environ["LLM_DRIFT_PAUSE"] = "enforce"
        # Empty cells
        paused, why = dm.is_drift_paused("trend", "1h")
        self.assertFalse(paused)
        self.assertIn("no_baselines", why)

    def test_enforce_clean_cell_does_not_pause(self) -> None:
        """Cell exists, no pause_count → don't halt."""
        os.environ["LLM_DRIFT_PAUSE"] = "enforce"
        np.random.seed(10)
        payload = _baseline_payload(np.random.normal(0, 1, 1000))
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        # Same-dist actuals → no drift
        actual = pd.DataFrame({"ofi_z": np.random.normal(0, 1, 1000)})
        actual.to_parquet(self.training_runs / "trend__1h.parquet")
        dm.run_once()
        paused, why = dm.is_drift_paused("trend", "1h")
        self.assertFalse(paused)
        self.assertEqual(why, "cell_clean")

    def test_enforce_paused_cell_blocks_trading(self) -> None:
        """Hard-feature pause severity + enforce → return (True, reason)."""
        os.environ["LLM_DRIFT_PAUSE"] = "enforce"
        np.random.seed(11)
        ref = np.random.normal(0, 1, 2000)
        payload = _baseline_payload(ref, feat="ofi_z")
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        # 5σ shift → pause
        actual = pd.DataFrame({"ofi_z": np.random.normal(5.0, 0.5, 2000)})
        actual.to_parquet(self.training_runs / "trend__1h.parquet")
        dm.run_once()
        paused, why = dm.is_drift_paused("trend", "1h")
        self.assertTrue(paused)
        self.assertIn("ofi_z", why)

    def test_enforce_unknown_cell_does_not_pause(self) -> None:
        os.environ["LLM_DRIFT_PAUSE"] = "enforce"
        # Some other (model, tf) trained but caller asks about a different one
        np.random.seed(12)
        payload = _baseline_payload(np.random.normal(0, 1, 500))
        (self.baselines / "trend__1h.json").write_text(json.dumps(payload))
        dm.run_once()
        paused, why = dm.is_drift_paused("base", "4h")  # not the trained cell
        self.assertFalse(paused)
        self.assertEqual(why, "cell_not_found")


class TestStartStop(_TmpRoot):
    def test_start_returns_true_first_time_false_after(self) -> None:
        # Make sure no previous test left the thread alive
        dm.stop()
        time.sleep(0.1)
        # Reset module-level handle
        dm._thread = None
        ok1 = dm.start(interval_s=3600)
        ok2 = dm.start(interval_s=3600)
        try:
            self.assertTrue(ok1, "first start should succeed")
            self.assertFalse(ok2, "second start should be idempotent no-op")
        finally:
            dm.stop()

    def test_disabled_env_returns_false(self) -> None:
        os.environ["DRIFT_MONITOR_DISABLED"] = "1"
        dm.stop()
        time.sleep(0.1)
        dm._thread = None
        try:
            ok = dm.start(interval_s=3600)
            self.assertFalse(ok, "DRIFT_MONITOR_DISABLED=1 should disable start")
        finally:
            os.environ.pop("DRIFT_MONITOR_DISABLED", None)


if __name__ == "__main__":
    unittest.main()
