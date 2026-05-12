"""F3 + F4 — Behavioral round-trip tests for critical dashboard API endpoints.

Every test uses app.test_client() to send a real HTTP request and asserts on
the response status code AND the payload structure.  No string-matching on
source text — all assertions invoke the actual route handlers.

F4 (orchestrator submit + dedup) is also covered here because the job
scheduler lives in app.py alongside the other endpoints.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dashboard.app import app  # noqa: E402


def _make_client():
    app.config["TESTING"] = True
    # DASHBOARD_API_KEY is empty in test env → require_api_key is a no-op.
    return app.test_client()


# ══════════════════════════════════════════════════════════════════════════════
# Helper base class
# ══════════════════════════════════════════════════════════════════════════════

class _DashboardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()
        self._ctx = self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def get(self, path: str) -> tuple[int, dict]:
        r = self._ctx.get(path)
        return r.status_code, r.get_json() or {}

    def post(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        r = self._ctx.post(
            path,
            data=json.dumps(body or {}),
            content_type="application/json",
        )
        return r.status_code, r.get_json() or {}


# ══════════════════════════════════════════════════════════════════════════════
# F3 — Critical read endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestStateEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/state")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/state")
        self.assertIsInstance(data, dict)


class TestControlGetEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/control")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/control")
        self.assertIsInstance(data, dict)


class TestControlPostEndpoint(_DashboardTestCase):
    def test_merge_field_returns_success(self) -> None:
        status, data = self.post("/api/control", {"running": True})
        self.assertEqual(status, 200)
        self.assertTrue(data.get("success"))
        self.assertIn("control", data)

    def test_merge_preserves_existing_fields(self) -> None:
        # Write initial state with two fields
        self.post("/api/control", {"field_a": "alpha", "field_b": "beta"})
        # Merge only field_a — field_b must survive
        _, data = self.post("/api/control", {"field_a": "updated"})
        ctrl = data.get("control", {})
        self.assertEqual(ctrl.get("field_a"), "updated")
        self.assertEqual(ctrl.get("field_b"), "beta")

    def test_non_dict_body_returns_400(self) -> None:
        r = self._ctx.post(
            "/api/control",
            data=json.dumps([1, 2, 3]),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)


class TestTradesEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/trades")
        self.assertEqual(status, 200)

    def test_returns_trades_key(self) -> None:
        _, data = self.get("/api/trades")
        self.assertIn("trades", data)
        self.assertIsInstance(data["trades"], list)


class TestMonitorHealthEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/monitor/health")
        self.assertEqual(status, 200)

    def test_has_bot_and_dash_keys(self) -> None:
        _, data = self.get("/api/monitor/health")
        self.assertIn("bot", data)
        self.assertIn("dash", data)

    def test_bot_entry_has_required_fields(self) -> None:
        _, data = self.get("/api/monitor/health")
        bot = data["bot"]
        self.assertIn("label", bot)
        self.assertIn("running", bot)


class TestPipelineStatusEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/pipeline/status")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/pipeline/status")
        self.assertIsInstance(data, dict)

    def test_process_alive_field_present(self) -> None:
        _, data = self.get("/api/pipeline/status")
        self.assertIn("process_alive", data)
        self.assertIsInstance(data["process_alive"], bool)


class TestTrainingJobsEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/training/jobs")
        self.assertEqual(status, 200)

    def test_has_jobs_and_total(self) -> None:
        _, data = self.get("/api/training/jobs")
        self.assertIn("jobs", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["jobs"], list)

    def test_limit_param_respected(self) -> None:
        r = self._ctx.get("/api/training/jobs?limit=2")
        data = r.get_json() or {}
        self.assertLessEqual(len(data.get("jobs", [])), 2)


class TestTrainingRulesEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/training/rules")
        self.assertEqual(status, 200)

    def test_has_ok_true(self) -> None:
        _, data = self.get("/api/training/rules")
        self.assertTrue(data.get("ok"))

    def test_has_matrix_key(self) -> None:
        _, data = self.get("/api/training/rules")
        self.assertIn("matrix", data)
        self.assertIsInstance(data["matrix"], list)

    def test_each_matrix_cell_has_required_fields(self) -> None:
        _, data = self.get("/api/training/rules")
        for cell in data.get("matrix", []):
            for field in ("model", "tf", "status"):
                self.assertIn(field, cell,
                              f"cell missing {field!r}: {cell}")


class TestStrategyFullEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/strategy/full")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/strategy/full")
        self.assertIsInstance(data, dict)


class TestModelsEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/models")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/models")
        self.assertIsInstance(data, dict)


class TestBalanceVirtualEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/balance/virtual")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/balance/virtual")
        self.assertIsInstance(data, dict)


class TestBacktestSummaryEndpoint(_DashboardTestCase):
    def test_returns_200(self) -> None:
        status, _ = self.get("/api/backtest/summary")
        self.assertEqual(status, 200)

    def test_returns_dict(self) -> None:
        _, data = self.get("/api/backtest/summary")
        self.assertIsInstance(data, dict)


# ══════════════════════════════════════════════════════════════════════════════
# F4 — Training job submit + dedup state-persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingJobSubmit(_DashboardTestCase):
    """Submit a training job via POST /api/training/run/<key> and verify the
    job ID is returned and visible in /api/training/jobs immediately.

    The actual trainer is patched out so no ML runs happen in tests.

    setUp/tearDown snapshot and restore _training_jobs so each test runs
    with the same in-memory state it started with (no bleed between tests).
    """

    def setUp(self) -> None:
        super().setUp()
        import src.dashboard.app as _app_mod
        with _app_mod._training_jobs_lock:
            self._jobs_snapshot = dict(_app_mod._training_jobs)

    def tearDown(self) -> None:
        import src.dashboard.app as _app_mod
        with _app_mod._training_jobs_lock:
            _app_mod._training_jobs.clear()
            _app_mod._training_jobs.update(self._jobs_snapshot)
        super().tearDown()

    def _patch_trainer(self):
        """Patch the cluster-dispatch thread so no real training happens."""
        import src.dashboard.app as _app_mod
        return patch.object(
            _app_mod, "_dispatch_training_to_cluster",
            side_effect=lambda job_id, key, n, tf, **kw: None,
        )

    def test_submit_known_key_returns_ok(self) -> None:
        # force=True bypasses dedup — we're testing the submit path, not dedup.
        with self._patch_trainer():
            status, data = self.post("/api/training/run/base", {"force": True})
        self.assertEqual(status, 200)
        self.assertTrue(data.get("ok"), f"expected ok=True, got: {data}")

    def test_submit_returns_job_id(self) -> None:
        with self._patch_trainer():
            _, data = self.post("/api/training/run/base", {"force": True})
        self.assertIn("job_id", data)
        self.assertIsInstance(data["job_id"], str)
        self.assertGreater(len(data["job_id"]), 0)

    def test_submitted_job_visible_in_jobs_list(self) -> None:
        with self._patch_trainer():
            _, submit_data = self.post("/api/training/run/trend", {"force": True})
        job_id = submit_data.get("job_id")
        self.assertIsNotNone(job_id)

        _, jobs_data = self.get("/api/training/jobs")
        job_ids = [j["job_id"] for j in jobs_data.get("jobs", [])]
        self.assertIn(job_id, job_ids,
                      f"submitted job {job_id} not found in /api/training/jobs")

    def test_submit_unknown_key_returns_400(self) -> None:
        status, data = self.post("/api/training/run/totally_fake_key_xyz", {})
        self.assertEqual(status, 400)
        self.assertFalse(data.get("ok"))
        self.assertIn("valid", data)

    def test_submit_invalid_tf_returns_400(self) -> None:
        # force=True bypasses dedup so tf validation runs (dedup fires first without force)
        with self._patch_trainer():
            status, data = self.post("/api/training/run/base", {"tf": "99x", "force": True})
        self.assertEqual(status, 400)

    def test_dedup_returns_409_on_duplicate(self) -> None:
        """Second submit for the same model (status=queued) must return 409.

        We inject a known queued job directly into _training_jobs so the
        test is deterministic — no race between the dispatch thread
        and the second HTTP request.
        """
        import src.dashboard.app as _app_mod
        import uuid

        fake_job_id = uuid.uuid4().hex[:12]
        with _app_mod._training_jobs_lock:
            _app_mod._training_jobs[fake_job_id] = {
                "job_id": fake_job_id,
                "model": "base",
                "status": "queued",
                "created_at": 0,
            }

        # Second submit for "base" without force — dedup must fire.
        with self._patch_trainer():
            status, data = self.post("/api/training/run/base", {"force": False})

        self.assertEqual(status, 409, f"expected 409, got {status}: {data}")
        self.assertFalse(data.get("ok"))
        self.assertEqual(data.get("error"), "already_running")
        self.assertIn("existing_job_id", data)

    def test_force_flag_bypasses_dedup(self) -> None:
        """force=True must submit a NEW job even if one is already queued."""
        import src.dashboard.app as _app_mod

        # Manually inject a fake queued job so dedup fires reliably.
        fake_job_id = "fakejob000001"
        with _app_mod._training_jobs_lock:
            _app_mod._training_jobs[fake_job_id] = {
                "job_id": fake_job_id,
                "model": "scalping",
                "status": "queued",
                "created_at": 0,
            }

        try:
            with self._patch_trainer():
                status, data = self.post(
                    "/api/training/run/scalping", {"force": True}
                )
            self.assertEqual(status, 200)
            self.assertTrue(data.get("ok"))
        finally:
            with _app_mod._training_jobs_lock:
                _app_mod._training_jobs.pop(fake_job_id, None)


class TestTrainingJobStatePersistence(_DashboardTestCase):
    """Verify _record_job writes into the in-memory store and that the
    /api/training/jobs endpoint reflects those mutations without a restart."""

    def test_job_fields_persisted_after_record(self) -> None:
        import time as _time
        import src.dashboard.app as _app_mod
        import uuid

        job_id = uuid.uuid4().hex[:12]
        # Use current time so this job is the newest and survives the 50-job cap.
        _app_mod._record_job(
            job_id,
            model="meta",
            n=1,
            tf="1h",
            status="queued",
            created_at=_time.time() + 9999,  # far future → always newest
        )
        try:
            _, data = self.get("/api/training/jobs?limit=50")
            ids_in_list = [j["job_id"] for j in data.get("jobs", [])]
            self.assertIn(job_id, ids_in_list)

            # Find our job
            our_job = next(j for j in data["jobs"] if j["job_id"] == job_id)
            self.assertEqual(our_job["model"], "meta")
            self.assertEqual(our_job["tf"], "1h")
            self.assertEqual(our_job["status"], "queued")
        finally:
            with _app_mod._training_jobs_lock:
                _app_mod._training_jobs.pop(job_id, None)

    def test_job_visible_in_list_after_record(self) -> None:
        """_record_job must make the job appear in /api/training/jobs immediately."""
        import time as _time
        import src.dashboard.app as _app_mod
        import uuid

        job_id = uuid.uuid4().hex[:12]
        _app_mod._record_job(
            job_id, model="rl", status="queued", created_at=_time.time() + 9998,
        )

        try:
            _, data = self.get("/api/training/jobs?limit=50")
            ids_in_list = [j["job_id"] for j in data.get("jobs", [])]
            self.assertIn(job_id, ids_in_list,
                          f"job {job_id} not found after _record_job; jobs: {ids_in_list[:5]}")
        finally:
            with _app_mod._training_jobs_lock:
                _app_mod._training_jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
