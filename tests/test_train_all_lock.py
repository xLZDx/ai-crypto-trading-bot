"""Phase 97 (2026-05-14) regression test — train_all_models single-instance
lock must serialize concurrent acquire attempts.

Pre-fix the lock had a TOCTOU race: `os.path.exists` + `open(,'r')` for
the CHECK and a separate `write_json` for the CLAIM. Two threads could
both pass the "no live owner" check before either wrote, and BOTH
acquired. This test proves the new transaction-based path serializes
them: exactly one acquires, others refuse.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestSingleInstanceLock(unittest.TestCase):
    """Verify the transaction-based lock refuses double-acquire while a
    live holder exists, and reclaims stale locks from dead PIDs.

    Mocks psutil.Process so the cmdline-check inside _acquire_run_lock
    matches our test process — pytest's cmdline contains "pytest" not
    "train_all_models", which would otherwise classify the previous
    holder as dead and reclaim the lock (defeating the test)."""

    def setUp(self) -> None:
        # Redirect the lock path to a per-test tmpdir so we don't fight
        # the real lock during regression.
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_lock = os.path.join(self.tmp.name, "train_all_models.lock")
        # Import the module fresh AND patch its _LOCK_PATH.
        from src.engine import train_all_models as tam
        self.tam = tam
        self._lp = mock.patch.object(tam, "_LOCK_PATH", self.tmp_lock)
        self._lp.start()
        # Make psutil's liveness probe think the current process is a
        # train_all_models.py — pytest's cmdline wouldn't otherwise match.
        # This is the only way to test the "lock held by alive holder"
        # branch from a pytest harness.
        self._proc_patch = mock.patch("psutil.Process")
        self._mock_proc_cls = self._proc_patch.start()
        fake = mock.MagicMock()
        fake.name.return_value = "python.exe"
        fake.cmdline.return_value = [
            "python.exe", "src/engine/train_all_models.py",
        ]
        self._mock_proc_cls.return_value = fake

    def tearDown(self) -> None:
        self._proc_patch.stop()
        self._lp.stop()
        self.tmp.cleanup()

    def test_first_acquire_succeeds(self) -> None:
        """No prior lock — first call wins."""
        self.assertTrue(self.tam._acquire_run_lock(force=False))
        # Lock file exists with our pid
        self.assertTrue(os.path.exists(self.tmp_lock))
        import json
        payload = json.loads(Path(self.tmp_lock).read_text())
        self.assertEqual(payload["pid"], os.getpid())

    def test_second_acquire_refuses_when_holder_alive(self) -> None:
        """First acquire wins; second acquire by the same live process refuses."""
        self.assertTrue(self.tam._acquire_run_lock(force=False))
        # Second call: same PID is alive (ourselves) → refuse without --force
        self.assertFalse(self.tam._acquire_run_lock(force=False))

    def test_force_reclaims_lock_held_by_alive_holder(self) -> None:
        """--force allows a second instance to proceed in parallel."""
        self.assertTrue(self.tam._acquire_run_lock(force=False))
        # Same PID, --force → True (warning logged about parallel run)
        self.assertTrue(self.tam._acquire_run_lock(force=True))

    def test_stale_lock_auto_reclaimed(self) -> None:
        """If the lock points at a dead PID, the next acquire reclaims it."""
        # Write a stale lock pointing at a nonexistent PID.
        import json
        Path(self.tmp_lock).write_text(json.dumps({
            "pid": 99_999_999,  # almost certainly dead
            "started_iso": "1999-01-01T00:00:00+00:00",
            "host": "ghost",
        }))
        # Should reclaim and return True
        self.assertTrue(self.tam._acquire_run_lock(force=False))
        # New lock points at our pid
        payload = json.loads(Path(self.tmp_lock).read_text())
        self.assertEqual(payload["pid"], os.getpid())

    def test_concurrent_acquire_serializes(self) -> None:
        """The TOCTOU regression: 2 threads racing to acquire — exactly
        one wins, the other refuses. Pre-fix both would have won."""
        # Reset baseline
        if os.path.exists(self.tmp_lock):
            os.remove(self.tmp_lock)

        results: list[bool] = []
        results_lock = threading.Lock()

        def attempt():
            ok = self.tam._acquire_run_lock(force=False)
            with results_lock:
                results.append(ok)

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Same-PID double-acquire: only ONE thread can claim the lock as
        # "new owner". The 3 others see "lock held by alive pid (us)" and
        # refuse. The transaction serializes them.
        ok_count = sum(1 for r in results if r)
        false_count = sum(1 for r in results if not r)
        self.assertEqual(ok_count + false_count, 4,
                         f"expected 4 total results, got {results}")
        self.assertEqual(ok_count, 1,
                         f"expected exactly 1 success; got {ok_count} (race not serialized): {results}")


if __name__ == "__main__":
    unittest.main()
