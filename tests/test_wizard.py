"""Phase 7 (2026-05-14) — Training Wizard tests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dashboard import wizard as wz  # noqa: E402


class TestRulebasedRecommender(unittest.TestCase):
    """Each rule produces the expected recommendation given a synthetic meta."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_models = Path(self.tmp.name) / "models"
        self.tmp_baselines = Path(self.tmp.name) / "baselines"
        self.tmp_models.mkdir(parents=True, exist_ok=True)
        self.tmp_baselines.mkdir(parents=True, exist_ok=True)
        self._mp = mock.patch.object(wz, "MODELS_DIR", self.tmp_models)
        self._bp = mock.patch.object(wz, "BASELINES_DIR", self.tmp_baselines)
        self._mp.start()
        self._bp.start()

    def tearDown(self) -> None:
        self._mp.stop()
        self._bp.stop()
        self.tmp.cleanup()

    def _write_meta(self, model: str, tf: str, payload: dict) -> Path:
        # Map back to the on-disk filename pattern.
        tmpl = wz.KNOWN_MODELS[model]
        path = self.tmp_models / tmpl.format(tf=tf)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_noise_floor_auc_produces_critical(self) -> None:
        self._write_meta("trend", "1h", {"auc": 0.51})
        rep = wz.suggest_for_model("trend", "1h")
        self.assertIsNotNone(rep.auc)
        critical = [r for r in rep.recommendations if r.severity == "critical"]
        self.assertGreaterEqual(len(critical), 1,
                                f"expected at least one critical for AUC=0.51, got {rep.recommendations}")
        self.assertTrue(any("noise floor" in r.title.lower() for r in critical))

    def test_below_target_auc_produces_high(self) -> None:
        self._write_meta("trend", "1h", {"auc": 0.53})
        rep = wz.suggest_for_model("trend", "1h")
        high = [r for r in rep.recommendations if r.severity == "high"]
        self.assertGreaterEqual(len(high), 1)

    def test_good_auc_no_critical_recommendation(self) -> None:
        self._write_meta("trend", "1h", {"auc": 0.62, "last_trained": "2026-05-14T00:00:00Z"})
        rep = wz.suggest_for_model("trend", "1h")
        critical = [r for r in rep.recommendations if r.severity == "critical"]
        self.assertEqual(len(critical), 0, f"unexpected critical: {critical}")

    def test_missing_cell_in_expected_matrix_flagged_high(self) -> None:
        # No meta file → wizard knows this cell is expected per matrix
        rep = wz.suggest_for_model("trend", "1h")
        missing = [r for r in rep.recommendations if "No trained" in r.title]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].severity, "high")

    def test_stale_training_flagged_medium(self) -> None:
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self._write_meta("trend", "1h", {"auc": 0.65, "last_trained": old})
        rep = wz.suggest_for_model("trend", "1h")
        stale = [r for r in rep.recommendations if "not retrained" in r.title]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, "medium")

    def test_drift_baseline_missing_flagged_medium(self) -> None:
        """Trained model but no drift baseline → flag as medium."""
        self._write_meta("trend", "1h", {"auc": 0.65, "last_trained": "2026-05-14T00:00:00Z"})
        rep = wz.suggest_for_model("trend", "1h")
        drift = [r for r in rep.recommendations if "Drift baseline missing" in r.title]
        self.assertGreaterEqual(len(drift), 1)

    def test_unknown_model_returns_no_recommendations(self) -> None:
        rep = wz.suggest_for_model("not-a-real-model", "1h")
        self.assertEqual(rep.recommendations, [])

    def test_report_to_dict_is_json_serializable(self) -> None:
        self._write_meta("trend", "1h", {"auc": 0.51})
        rep = wz.suggest_for_model("trend", "1h")
        d = rep.to_dict()
        json.dumps(d)  # must serialize cleanly
        self.assertEqual(d["model"], "trend")
        self.assertEqual(d["tf"], "1h")
        self.assertIn("recommendations", d)

    def test_recommendations_ranked_by_severity(self) -> None:
        # AUC = 0.51 (critical) + missing baseline (medium); critical should come first.
        self._write_meta("trend", "1h", {"auc": 0.51, "last_trained": "2026-05-14T00:00:00Z"})
        rep = wz.suggest_for_model("trend", "1h")
        sevs = [r.severity for r in rep.recommendations]
        # critical should appear before medium
        if "critical" in sevs and "medium" in sevs:
            self.assertLess(sevs.index("critical"), sevs.index("medium"))


class TestAskLLM(unittest.TestCase):
    """ask_llm: empty question handling + no-API-key path."""

    def test_empty_question_returns_friendly_message(self) -> None:
        r = wz.ask_llm("")
        self.assertEqual(r["source"], "empty")

    def test_no_api_key_returns_friendly_message(self) -> None:
        """When GEMINI_API_KEY is unset, ask_llm returns a friendly note
        rather than crashing the wizard endpoint."""
        # Mock AgenticLLM to report inactive (the no-API-key state).
        with mock.patch("src.engine.agentic_llm.AgenticLLM") as MockLLM:
            inst = MockLLM.return_value
            inst.is_active = False
            r = wz.ask_llm("why is trend underperforming?")
        self.assertEqual(r["source"], "no_api_key")
        self.assertIn("GEMINI_API_KEY", r["answer"])


if __name__ == "__main__":
    unittest.main()
