"""Phase D (2026-05-14) — training_history regression tests."""
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

from src.analytics import training_history as th  # noqa: E402


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.history_path = Path(self.tmp.name) / "training_runs_history.json"
        self._patch = mock.patch.object(th, "HISTORY_PATH", self.history_path)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self.tmp.cleanup()


class TestScoreRun(unittest.TestCase):
    def test_all_three_metrics(self) -> None:
        s = th.score_run({"accuracy_test": 60.0, "auc_roc": 0.6, "win_precision": 70.0})
        # 0.5*60 + 0.3*60 + 0.2*70 = 62
        self.assertAlmostEqual(s, 62.0, places=3)

    def test_normalizes_fraction_accuracy(self) -> None:
        # 0.65 should be treated as 65%
        s = th.score_run({"accuracy_test": 65.0})  # already percent
        self.assertAlmostEqual(s, 65.0, places=3)

    def test_no_metrics(self) -> None:
        self.assertIsNone(th.score_run({}))
        self.assertIsNone(th.score_run(None))

    def test_falls_back_to_win_rate(self) -> None:
        s = th.score_run({"accuracy_test": 60, "win_rate_pct": 70})
        # 0.5*60 + 0.2*70 / 0.7 = (30+14)/0.7 = 62.857
        self.assertAlmostEqual(s, (30.0 + 14.0) / 0.7, places=3)


class TestRecordAndQuery(_Tmp):
    def test_first_run_becomes_baseline(self) -> None:
        rid = th.record_run("trend", "1h",
                            metrics={"accuracy_test": 60, "auc_roc": 0.6})
        rows = th.get_runs()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], rid)
        self.assertTrue(rows[0]["is_baseline"])
        self.assertIsNone(rows[0]["delta_vs_baseline"])

    def test_second_run_has_delta_vs_baseline(self) -> None:
        r1 = th.record_run("trend", "1h",
                           metrics={"accuracy_test": 60, "auc_roc": 0.6})
        r2 = th.record_run("trend", "1h",
                           metrics={"accuracy_test": 65, "auc_roc": 0.65})
        rows = th.get_runs(model="trend", tf="1h")
        # rows are newest-first
        latest = rows[0]
        first  = rows[1]
        self.assertEqual(latest["run_id"], r2)
        self.assertEqual(first["run_id"], r1)
        self.assertFalse(latest["is_baseline"])
        self.assertTrue(first["is_baseline"])
        d = latest["delta_vs_baseline"]
        self.assertAlmostEqual(d["d_accuracy_test"], 5.0, places=3)
        self.assertAlmostEqual(d["d_auc_roc"], 0.05, places=3)
        # Score delta = score(r2) - score(r1)
        self.assertGreater(d["d_score"], 0)

    def test_different_cells_have_separate_baselines(self) -> None:
        th.record_run("trend", "1h", metrics={"accuracy_test": 60})
        th.record_run("base", "4h", metrics={"accuracy_test": 55})
        self.assertIsNotNone(th.get_baseline("trend", "1h"))
        self.assertIsNotNone(th.get_baseline("base", "4h"))
        self.assertIsNone(th.get_baseline("scalping", "1m"))

    def test_get_runs_filters(self) -> None:
        th.record_run("trend", "1h", metrics={"accuracy_test": 60})
        th.record_run("base", "1h", metrics={"accuracy_test": 55})
        self.assertEqual(len(th.get_runs(model="trend")), 1)
        self.assertEqual(len(th.get_runs(tf="1h")), 2)
        self.assertEqual(len(th.get_runs(model="base", tf="1h")), 1)

    def test_get_runs_limit(self) -> None:
        for i in range(5):
            th.record_run("trend", "1h", metrics={"accuracy_test": 60 + i})
        self.assertEqual(len(th.get_runs(model="trend", limit=3)), 3)


class TestPromoteBaseline(_Tmp):
    def test_promote_changes_baseline_and_recomputes_deltas(self) -> None:
        r1 = th.record_run("trend", "1h", metrics={"accuracy_test": 60})
        r2 = th.record_run("trend", "1h", metrics={"accuracy_test": 65})
        r3 = th.record_run("trend", "1h", metrics={"accuracy_test": 70})
        # Before promote: r1 is baseline; r3 delta vs r1 = +10
        rows = th.get_runs(model="trend", tf="1h")
        latest = rows[0]
        self.assertEqual(latest["run_id"], r3)
        self.assertAlmostEqual(latest["delta_vs_baseline"]["d_accuracy_test"], 10.0)
        # Promote r2 as the baseline.
        ok = th.promote_baseline(r2)
        self.assertTrue(ok)
        rows2 = th.get_runs(model="trend", tf="1h")
        # r2 is now baseline (delta=None), r3 delta vs r2 = +5, r1 delta vs r2 = -5
        for r in rows2:
            if r["run_id"] == r2:
                self.assertTrue(r["is_baseline"])
                self.assertIsNone(r["delta_vs_baseline"])
            elif r["run_id"] == r3:
                self.assertAlmostEqual(r["delta_vs_baseline"]["d_accuracy_test"], 5.0)
            elif r["run_id"] == r1:
                self.assertAlmostEqual(r["delta_vs_baseline"]["d_accuracy_test"], -5.0)

    def test_promote_unknown_run_id_fails(self) -> None:
        th.record_run("trend", "1h", metrics={"accuracy_test": 60})
        self.assertFalse(th.promote_baseline("bogus_run_id"))


class TestWinningHyperparameters(_Tmp):
    def test_returns_highest_score_hp(self) -> None:
        th.record_run("trend", "1h",
                      metrics={"accuracy_test": 60}, hp={"n_estimators": 100})
        th.record_run("trend", "1h",
                      metrics={"accuracy_test": 65}, hp={"n_estimators": 300})
        th.record_run("trend", "1h",
                      metrics={"accuracy_test": 62}, hp={"n_estimators": 200})
        winning = th.winning_hyperparameters("trend", "1h")
        self.assertIsNotNone(winning)
        self.assertEqual(winning["best_hp"]["n_estimators"], 300)
        self.assertEqual(winning["n_runs_considered"], 3)

    def test_returns_none_when_no_runs(self) -> None:
        self.assertIsNone(th.winning_hyperparameters("trend", "1h"))


class TestBackfill(_Tmp):
    def test_backfill_idempotent(self) -> None:
        # Build a fake models/ dir with one meta JSON, point MODELS_DIR there.
        fake_models = Path(self.tmp.name) / "models"
        fake_models.mkdir()
        meta = {
            "model": "Trend RF",
            "accuracy": 65.0,
            "auc_roc": 0.6,
            "long_accuracy": 0.0,
            "short_accuracy": 65.0,
            "n_features": 22,
            "n_samples": 100_000,
            "last_trained": "2026-05-14T12:36:39+00:00",
        }
        (fake_models / "trend_model_meta.json").write_text(json.dumps(meta))
        with mock.patch.object(th, "MODELS_DIR", fake_models):
            n1 = th.backfill_from_meta_files()
            n2 = th.backfill_from_meta_files()
        self.assertEqual(n1, 1, "first backfill should add 1 row")
        self.assertEqual(n2, 0, "second backfill should be idempotent")

    def test_backfill_picks_up_per_tf_metas(self) -> None:
        fake_models = Path(self.tmp.name) / "models"
        fake_models.mkdir()
        # Per-TF: trend_4h_meta.json -> ('trend', '4h')
        (fake_models / "trend_4h_meta.json").write_text(json.dumps({
            "accuracy": 55.0, "n_features": 20,
            "last_trained": "2026-05-12T21:19:32+00:00",
        }))
        with mock.patch.object(th, "MODELS_DIR", fake_models):
            n = th.backfill_from_meta_files()
        self.assertEqual(n, 1)
        rows = th.get_runs(model="trend", tf="4h")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_features"], 20)


if __name__ == "__main__":
    unittest.main()
