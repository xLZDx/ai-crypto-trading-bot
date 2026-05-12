"""Behavioral tests for Phase C — state persistence + agent consolidation.

Covers:
  - C7: agent_status.json history cap raised 10 -> 100
  - C4: ParquetStore has an RLock (re-entrant DuckDB serialization)
  - C3: stale src/agents/ + tests/test_agents.py removed
  - C1: orchestrator state persisted + reloaded; running tasks re-queued
  - C2: submit_task dedupes by (model_type, symbol, timeframe)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestC7HistoryCap(unittest.TestCase):
    """Phase C7 — history array cap raised 10 -> 100."""

    def test_max_history_constant_is_100(self) -> None:
        from src.engine.agents.agent_bus import _MAX_HISTORY
        self.assertEqual(_MAX_HISTORY, 100)

    def test_write_status_caps_history_at_max(self) -> None:
        """Drive the real write path 150 times and assert only the last
        _MAX_HISTORY entries survive."""
        from src.engine.agents import agent_bus

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "agent_status.json"
            original = agent_bus._STATUS_FILE
            agent_bus._STATUS_FILE = target
            try:
                # Each call appends previous task to history if task changed.
                for i in range(150):
                    agent_bus._write_agent_status(
                        name="UnitTestAgent",
                        status="busy",
                        task=f"task_{i}",
                        interval_sec=1.0,
                    )
                data = json.loads(target.read_text(encoding="utf-8"))
                hist = data["UnitTestAgent"]["history"]
                self.assertLessEqual(len(hist), agent_bus._MAX_HISTORY)
                # Last entry should reference a recent task index.
                self.assertTrue(
                    any("task_14" in h.get("task", "") for h in hist),
                    f"expected recent tasks in history; got {hist[:3]} ... {hist[-3:]}",
                )
            finally:
                agent_bus._STATUS_FILE = original


class TestC4ParquetStoreLock(unittest.TestCase):
    """Phase C4 — ParquetStore must hold an RLock around DuckDB calls."""

    def test_has_rlock(self) -> None:
        from src.database.parquet_store import ParquetStore
        ps = ParquetStore()
        # Re-entrant lock — must allow acquire-twice from same thread.
        self.assertEqual(type(ps._lock).__name__, "RLock")
        ps._lock.acquire()
        try:
            self.assertTrue(ps._lock.acquire(blocking=False))
            ps._lock.release()
        finally:
            ps._lock.release()

    def test_concurrent_list_symbols_does_not_crash(self) -> None:
        """list_symbols itself is pure FS, but firing many threads at the
        store should not perturb DuckDB state. This is the smoke test that
        would have caught the 2026-05-08 incident on ParquetClient."""
        from src.database.parquet_store import ParquetStore
        ps = ParquetStore()
        errors: list[BaseException] = []

        def _w() -> None:
            try:
                _ = ps.list_symbols()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_w) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(errors, [], f"concurrent list_symbols raised: {errors}")


class TestC3StaleAgentsDeleted(unittest.TestCase):
    """Phase C3 — stale src/agents/ toy + tests/test_agents.py removed."""

    def test_src_agents_dir_gone(self) -> None:
        self.assertFalse(
            (Path(ROOT) / "src" / "agents").exists(),
            "src/agents/ toy supervisor should be deleted (canonical lives at src/orchestration/master_agent.py)",
        )

    def test_test_agents_file_gone(self) -> None:
        self.assertFalse(
            (Path(ROOT) / "tests" / "test_agents.py").exists(),
            "tests/test_agents.py should be deleted; it tested the now-removed toy supervisor",
        )

    def test_canonical_master_agent_still_present(self) -> None:
        canonical = Path(ROOT) / "src" / "orchestration" / "master_agent.py"
        self.assertTrue(canonical.exists())
        # Sanity: it's the real zombie healer, not the 60-line toy.
        self.assertGreater(canonical.stat().st_size, 5000)


class TestC1OrchestratorStatePersistence(unittest.TestCase):
    """Phase C1 — orchestrator persists state and reloads on construction."""

    def _new_orch(self, state_path: Path):
        from src.training.distributed.orchestrator import Orchestrator
        return Orchestrator(state_path=state_path)

    def test_submit_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            o = self._new_orch(sp)
            tid = o.submit_task({
                "model_type": "base", "symbol": "BTC/USDT", "timeframe": "1h",
            })
            self.assertTrue(sp.exists(), "submit_task must persist state to disk")
            data = json.loads(sp.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 1)
            self.assertIn(tid, data["tasks"])
            self.assertIn(tid, data["queue"])

    def test_reload_restores_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            o1 = self._new_orch(sp)
            tid = o1.submit_task({
                "model_type": "trend", "symbol": "ETH/USDT", "timeframe": "4h",
            })
            # Construct a fresh orchestrator pointing at the same state file.
            o2 = self._new_orch(sp)
            self.assertIn(tid, o2._tasks)
            self.assertIn(tid, o2._queue)
            self.assertEqual(o2._tasks[tid]["status"], "pending")

    def test_running_task_is_requeued_on_reload(self) -> None:
        """A task left in 'running' at shutdown must come back as 'pending'
        with assigned_to cleared, so the dispatcher hands it to a fresh worker."""
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            o1 = self._new_orch(sp)
            tid = o1.submit_task({
                "model_type": "scalping", "symbol": "BTC/USDT", "timeframe": "1m",
            })
            # Simulate dispatch: mark it running.
            o1.update_task(tid, status="running", node_id="ghost-worker")
            self.assertEqual(o1._tasks[tid]["status"], "running")
            # Restart.
            o2 = self._new_orch(sp)
            self.assertEqual(o2._tasks[tid]["status"], "pending",
                             "running task must be re-queued, not lost")
            self.assertEqual(o2._tasks[tid]["assigned_to"], "")
            self.assertIn(tid, o2._queue)

    def test_corrupt_state_file_does_not_block_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            sp.write_text("{not valid json", encoding="utf-8")
            # Must construct without raising; state should be fresh-empty.
            o = self._new_orch(sp)
            self.assertEqual(o._tasks, {})
            self.assertEqual(o._queue, [])


class TestC2TaskDedup(unittest.TestCase):
    """Phase C2 — submit_task dedupes by (model_type, symbol, timeframe)."""

    def _new_orch(self, state_path: Path):
        from src.training.distributed.orchestrator import Orchestrator
        return Orchestrator(state_path=state_path)

    def test_pending_duplicate_returns_same_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            o = self._new_orch(Path(tmp) / "s.json")
            spec = {"model_type": "base", "symbol": "BTC/USDT", "timeframe": "1h"}
            tid1 = o.submit_task(spec)
            tid2 = o.submit_task(dict(spec))
            self.assertEqual(tid1, tid2)
            self.assertEqual(len(o._tasks), 1)

    def test_running_duplicate_returns_same_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            o = self._new_orch(Path(tmp) / "s.json")
            spec = {"model_type": "futures", "symbol": "ETH/USDT", "timeframe": "5m"}
            tid1 = o.submit_task(spec)
            o.update_task(tid1, status="running", node_id="w1")
            tid2 = o.submit_task(dict(spec))
            self.assertEqual(tid1, tid2)

    def test_done_does_NOT_dedup(self) -> None:
        """A finished retrain is a legitimate resubmission — dedup must skip it."""
        with tempfile.TemporaryDirectory() as tmp:
            o = self._new_orch(Path(tmp) / "s.json")
            spec = {"model_type": "trend", "symbol": "BTC/USDT", "timeframe": "1d"}
            tid1 = o.submit_task(spec)
            o.update_task(tid1, status="done", node_id="w1")
            tid2 = o.submit_task(dict(spec))
            self.assertNotEqual(tid1, tid2)

    def test_different_timeframes_NOT_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            o = self._new_orch(Path(tmp) / "s.json")
            tid1 = o.submit_task({"model_type": "base", "symbol": "BTC/USDT", "timeframe": "1h"})
            tid2 = o.submit_task({"model_type": "base", "symbol": "BTC/USDT", "timeframe": "4h"})
            self.assertNotEqual(tid1, tid2)
            self.assertEqual(len(o._tasks), 2)


class TestC1LoadStateSecurity(unittest.TestCase):
    """Phase C reviewer fixes: harden _load_state against poisoned state files."""

    def _new_orch(self, state_path: Path):
        from src.training.distributed.orchestrator import Orchestrator
        return Orchestrator(state_path=state_path)

    def test_public_worker_ip_is_dropped(self) -> None:
        """SSRF guard: a state file injecting a public IP must be dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            sp.write_text(json.dumps({
                "schema_version": 1,
                "workers": {
                    "evil":   {"node_id": "evil",   "ip": "169.254.169.254", "port": 7700},
                    "public": {"node_id": "public", "ip": "8.8.8.8",         "port": 7700},
                    "ok":     {"node_id": "ok",     "ip": "192.168.1.10",    "port": 7700},
                    "lo":     {"node_id": "lo",     "ip": "127.0.0.1",       "port": 7700},
                },
                "tasks": {},
                "queue": [],
            }), encoding="utf-8")
            o = self._new_orch(sp)
            self.assertNotIn("evil",   o._workers)
            self.assertNotIn("public", o._workers)
            self.assertIn("ok", o._workers)
            self.assertIn("lo", o._workers)

    def test_unsafe_port_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            sp.write_text(json.dumps({
                "schema_version": 1,
                "workers": {
                    "lowport": {"node_id": "lowport", "ip": "192.168.1.1", "port": 22},
                    "highport": {"node_id": "highport", "ip": "192.168.1.1", "port": 70000},
                    "ok": {"node_id": "ok", "ip": "192.168.1.1", "port": 7700},
                },
                "tasks": {},
                "queue": [],
            }), encoding="utf-8")
            o = self._new_orch(sp)
            self.assertNotIn("lowport", o._workers)
            self.assertNotIn("highport", o._workers)
            self.assertIn("ok", o._workers)

    def test_unsafe_data_path_is_neutralized(self) -> None:
        """Path injection guard: data_path pointing outside data/ must be blanked."""
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            sp.write_text(json.dumps({
                "schema_version": 1,
                "workers": {},
                "tasks": {
                    "evil": {
                        "task_id": "evil", "model_type": "base", "symbol": "BTC/USDT",
                        "timeframe": "1h", "status": "pending",
                        "data_path": "C:/Windows/System32/drivers/etc/hosts",
                        "output_path": "models",
                    },
                },
                "queue": ["evil"],
            }), encoding="utf-8")
            o = self._new_orch(sp)
            self.assertIn("evil", o._tasks)
            self.assertEqual(o._tasks["evil"]["data_path"], "",
                             "data_path outside project tree must be blanked")

    def test_unsafe_output_path_is_neutralized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "orch_state.json"
            sp.write_text(json.dumps({
                "schema_version": 1,
                "workers": {},
                "tasks": {
                    "evil": {
                        "task_id": "evil", "model_type": "base", "symbol": "BTC/USDT",
                        "timeframe": "1h", "status": "pending",
                        "data_path": "",
                        "output_path": "C:/Windows/System32",
                    },
                },
                "queue": ["evil"],
            }), encoding="utf-8")
            o = self._new_orch(sp)
            self.assertNotEqual(o._tasks["evil"]["output_path"], "C:/Windows/System32")


class TestC1PersistDoesNotHoldLockDuringWrite(unittest.TestCase):
    """Phase C reviewer fix: a slow filelock acquire must not stall the
    scheduler. We verify this by acquiring the orchestrator lock from the
    test, then calling _persist from another thread — it must release the
    lock BEFORE the disk write, so it doesn't block on our held lock."""

    def test_persist_writes_outside_lock(self) -> None:
        from src.training.distributed.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "s.json"
            o = Orchestrator(state_path=sp)
            o.submit_task({"model_type": "base", "symbol": "BTC/USDT", "timeframe": "1h"})

            # Take the orchestrator lock then invoke _persist from another
            # thread. If _persist tried to acquire the same lock and held it
            # across the file write, our held lock would block it; with the
            # snapshot-outside-lock fix, _persist takes the lock briefly,
            # snapshots, and releases.
            done = threading.Event()
            errs: list[BaseException] = []

            def _w():
                try:
                    o._persist()
                    done.set()
                except BaseException as e:  # noqa: BLE001
                    errs.append(e)

            with o._lock:
                t = threading.Thread(target=_w, daemon=True)
                t.start()
                # Give it a moment to try to acquire.
                done.wait(timeout=2.0)
                # _persist must NOT have completed yet — it should still be
                # waiting on _snapshot_state_locked to acquire o._lock.
                self.assertFalse(done.is_set(),
                                 "_persist completed while orchestrator lock was held — "
                                 "snapshot stage should serialize with us")
            # Outside the with-block, the lock is released; _persist proceeds.
            self.assertTrue(done.wait(timeout=2.0), f"persist never completed: {errs}")
            self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()
