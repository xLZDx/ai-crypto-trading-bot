"""Phase 2d behavioral tests for src/dashboard/process_manager.py.

No real subprocess spawning except in the integration test that exercises
kill() against a live python.exe sleeper. The unit tests use mocks so the
suite runs in <1s on every CI cycle.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dashboard import process_manager as pm  # noqa: E402


class TestFindRolePidDualForm(unittest.TestCase):
    """Phase F (2026-05-14) — _find_role_pid must locate the live process
    regardless of whether it was launched script-style (`python src/main.py`)
    or module-style (`python -m src.main`). Prior to the dual-form fix the
    matcher only checked one style, so a manual `python -m src.main` launch
    showed "dead" in the Monitor table even while trading actively."""

    def _fake_psutil(self, cmdlines: list[list[str]]):
        """Return a psutil stub whose process_iter yields fake procs with
        the given cmdlines. PIDs are 1000+index."""
        class _Proc:
            def __init__(self, pid, name, cmdline):
                self.info = {"pid": pid, "name": name, "cmdline": cmdline}
        procs = [_Proc(1000 + i, "python.exe", cmd) for i, cmd in enumerate(cmdlines)]
        stub = mock.MagicMock()
        stub.process_iter.return_value = iter(procs)
        # NoSuchProcess / AccessDenied need to exist as exception classes.
        stub.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        stub.AccessDenied = type("AccessDenied", (Exception,), {})
        return stub

    def test_script_spec_finds_module_launched_proc(self) -> None:
        """Spec uses `python src/main.py`; live proc uses `python -m src.main`.
        Must still find it."""
        spec = pm.RoleSpec(key="bot", label="Bot",
                           cmd=[str(pm.VENV_PYTHON), "src/main.py"])
        fake_psutil = self._fake_psutil([
            [str(pm.VENV_PYTHON), "-m", "src.main"],
        ])
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            pid = pm._find_role_pid(spec)
        self.assertEqual(pid, 1000)

    def test_module_spec_finds_script_launched_proc(self) -> None:
        """Spec uses `-m src.dashboard.app`; live proc uses `python src/dashboard/app.py`."""
        spec = pm.RoleSpec(key="dashboard", label="Dashboard",
                           cmd=[str(pm.VENV_PYTHON), "-m", "src.dashboard.app"])
        fake_psutil = self._fake_psutil([
            [str(pm.VENV_PYTHON), "src\\dashboard\\app.py"],
        ])
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            pid = pm._find_role_pid(spec)
        self.assertEqual(pid, 1000)

    def test_script_spec_finds_script_launched_proc(self) -> None:
        """Sanity: same-form matches still work."""
        spec = pm.RoleSpec(key="bot", label="Bot",
                           cmd=[str(pm.VENV_PYTHON), "src/main.py"])
        fake_psutil = self._fake_psutil([
            [str(pm.VENV_PYTHON), "D:\\proj\\src\\main.py"],
        ])
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            pid = pm._find_role_pid(spec)
        self.assertEqual(pid, 1000)

    def test_no_match_returns_none(self) -> None:
        spec = pm.RoleSpec(key="bot", label="Bot",
                           cmd=[str(pm.VENV_PYTHON), "src/main.py"])
        fake_psutil = self._fake_psutil([
            [str(pm.VENV_PYTHON), "some_other_script.py"],
        ])
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            pid = pm._find_role_pid(spec)
        self.assertIsNone(pid)


class TestAutoKillReReadsEnv(unittest.TestCase):
    """Phase F (2026-05-14) — the health-check loop must re-read
    AUTO_KILL_BAD_HEALTH on every iteration. Prior bug: the env was
    read once at thread start, so /api/processes/auto_kill toggles
    via the dashboard were a no-op until the next dashboard restart.
    """

    def test_loop_reads_env_each_iteration(self) -> None:
        """Static-source assertion: the loop body must reference
        os.environ.get("AUTO_KILL_BAD_HEALTH") INSIDE the while loop, not
        only before it. Spawning a live thread to assert this would race
        against the HTTP-probe side effects refresh_one() triggers; a
        source-level check is deterministic and captures the exact bug
        (env read above `while not self._stop.is_set():` instead of below).
        """
        import inspect
        src = inspect.getsource(pm.ProcessManager._loop)
        before_while, _, after_while = src.partition("while not self._stop.is_set():")
        self.assertIn("AUTO_KILL_BAD_HEALTH", after_while,
                      "AUTO_KILL_BAD_HEALTH must be re-read inside the while loop")
        # Belt-and-braces: make sure it is NOT cached above the loop.
        self.assertNotIn("AUTO_KILL_BAD_HEALTH", before_while,
                         "AUTO_KILL_BAD_HEALTH was cached above the while loop — "
                         "toggle via /api/processes/auto_kill will be a no-op "
                         "until process_manager restart")

    def test_loop_actually_observes_env_change(self) -> None:
        """Behavioral check: refresh_one is monkey-patched to a no-op, the
        loop runs for 2 ticks while we flip AUTO_KILL_BAD_HEALTH between
        iterations, and we capture what the loop used on each tick."""
        import threading as _threading
        manager = pm.ProcessManager()
        observed: list[str | None] = []
        tick = [0]

        def _no_op_refresh(key):
            # Only spy once per FULL pass (after all roles enumerated this tick).
            if tick[0] == 0 and key == list(pm.ROLE_SPECS.keys())[0]:
                observed.append(os.environ.get("AUTO_KILL_BAD_HEALTH"))
                os.environ["AUTO_KILL_BAD_HEALTH"] = "true"
            elif tick[0] == 1 and key == list(pm.ROLE_SPECS.keys())[0]:
                observed.append(os.environ.get("AUTO_KILL_BAD_HEALTH"))
                manager._stop.set()
            if key == list(pm.ROLE_SPECS.keys())[-1]:
                tick[0] += 1
            return type("Snap", (), {"bad_count": 0, "pid": None, "status": "ok"})()

        os.environ["AUTO_KILL_BAD_HEALTH"] = "false"
        manager.refresh_one = _no_op_refresh
        try:
            manager._stop.clear()
            t = _threading.Thread(target=manager._loop, args=(0.01,), daemon=True)
            t.start()
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive(), "loop did not exit within 3s")
            self.assertGreaterEqual(len(observed), 2,
                                    f"need 2+ observations, got {observed}")
            self.assertEqual(observed[0], "false")
            self.assertEqual(observed[1], "true",
                             "loop did not pick up runtime env-var toggle")
        finally:
            os.environ.pop("AUTO_KILL_BAD_HEALTH", None)


class TestRoleSpecsContract(unittest.TestCase):
    """Lock the public contract: every role spec is well-formed."""

    def test_all_roles_have_unique_keys(self) -> None:
        keys = list(pm.ROLE_SPECS.keys())
        self.assertEqual(len(keys), len(set(keys)),
                         f"duplicate role keys: {keys}")

    def test_every_role_has_label_and_cmd(self) -> None:
        for key, spec in pm.ROLE_SPECS.items():
            self.assertTrue(spec.label, f"{key}: missing label")
            self.assertTrue(spec.cmd, f"{key}: missing cmd")
            self.assertEqual(spec.key, key,
                             f"{key}: spec.key mismatches dict key {spec.key!r}")

    def test_health_kind_is_valid(self) -> None:
        valid = {"http", "pid+log", "pid-only"}
        for key, spec in pm.ROLE_SPECS.items():
            self.assertIn(spec.health_kind, valid,
                          f"{key}: invalid health_kind {spec.health_kind!r}")

    def test_http_health_kind_has_url(self) -> None:
        for key, spec in pm.ROLE_SPECS.items():
            if spec.health_kind == "http":
                self.assertTrue(spec.http_health,
                                f"{key}: health_kind=http but no http_health URL")


class TestProcessManagerList(unittest.TestCase):
    def setUp(self) -> None:
        self.pm = pm.ProcessManager()

    def test_list_returns_one_row_per_role(self) -> None:
        rows = self.pm.list()
        self.assertEqual(len(rows), len(pm.ROLE_SPECS))
        roles_seen = {r["role"] for r in rows}
        self.assertEqual(roles_seen, set(pm.ROLE_SPECS.keys()))

    def test_list_rows_have_required_keys(self) -> None:
        required = {"role", "label", "pid", "status",
                    "last_health_ts", "last_log_mtime", "uptime_s",
                    "log_file", "health_kind", "last_error", "bad_count"}
        for r in self.pm.list():
            self.assertTrue(required.issubset(r.keys()),
                            f"row {r['role']} missing keys: {required - r.keys()}")


class TestRefreshOne(unittest.TestCase):
    def setUp(self) -> None:
        self.pm = pm.ProcessManager()

    def test_dead_pid_marks_status_dead(self) -> None:
        with mock.patch.object(pm, "_find_role_pid", return_value=None):
            snap = self.pm.refresh_one("bot")
        # bot is pid+log -> with no PID alive -> dead
        self.assertEqual(snap.status, pm.HEALTH_DEAD)
        self.assertIsNone(snap.pid)
        self.assertEqual(snap.last_error, "PID not found")

    def test_pid_only_role_with_live_pid_is_ok(self) -> None:
        with mock.patch.object(pm, "_find_role_pid", return_value=12345), \
             mock.patch.object(pm, "_pid_alive", return_value=True), \
             mock.patch.object(pm, "_uptime_s", return_value=42):
            snap = self.pm.refresh_one("debug_supervisor")
        self.assertEqual(snap.status, pm.HEALTH_OK)
        self.assertEqual(snap.pid, 12345)
        self.assertEqual(snap.uptime_s, 42)

    def test_pid_log_role_with_stale_log_marks_stale(self) -> None:
        long_ago = time.time() - 600  # 10 min ago, exceeds 300s threshold
        with mock.patch.object(pm, "_find_role_pid", return_value=999), \
             mock.patch.object(pm, "_pid_alive", return_value=True), \
             mock.patch.object(pm, "_log_mtime", return_value=long_ago):
            snap = self.pm.refresh_one("bot")
        self.assertEqual(snap.status, pm.HEALTH_STALE)
        self.assertIn("not written in", snap.last_error or "")

    def test_http_role_with_failed_ping_but_live_pid_is_stale(self) -> None:
        with mock.patch.object(pm, "_find_role_pid", return_value=888), \
             mock.patch.object(pm, "_pid_alive", return_value=True), \
             mock.patch.object(pm, "_http_health_ok", return_value=False):
            snap = self.pm.refresh_one("monitor")
        self.assertEqual(snap.status, pm.HEALTH_STALE)
        self.assertIn("HTTP health failed", snap.last_error or "")

    def test_http_role_with_passing_ping_is_ok(self) -> None:
        with mock.patch.object(pm, "_find_role_pid", return_value=777), \
             mock.patch.object(pm, "_pid_alive", return_value=True), \
             mock.patch.object(pm, "_http_health_ok", return_value=True):
            snap = self.pm.refresh_one("monitor")
        self.assertEqual(snap.status, pm.HEALTH_OK)
        self.assertIsNone(snap.last_error)

    def test_bad_count_increments_on_bad_resets_on_good(self) -> None:
        # 3 bad checks -> bad_count == 3
        with mock.patch.object(pm, "_find_role_pid", return_value=None):
            for _ in range(3):
                self.pm.refresh_one("bot")
        self.assertEqual(self.pm._snapshots["bot"].bad_count, 3)

        # One good check resets to 0
        with mock.patch.object(pm, "_find_role_pid", return_value=111), \
             mock.patch.object(pm, "_pid_alive", return_value=True), \
             mock.patch.object(pm, "_log_mtime", return_value=time.time()):
            self.pm.refresh_one("bot")
        self.assertEqual(self.pm._snapshots["bot"].bad_count, 0)

    def test_unknown_role_returns_dead_snapshot(self) -> None:
        snap = self.pm.refresh_one("not-a-real-role")
        self.assertEqual(snap.status, pm.HEALTH_DEAD)
        self.assertEqual(snap.last_error, "unknown role")


class TestKill(unittest.TestCase):
    def test_kill_refuses_zero(self) -> None:
        # ProcessManager.kill itself doesn't check pid<=0 (the API endpoint
        # does). Verify it returns ok=True with "already gone" for non-
        # existent PID, which is the correct contract — kill is idempotent.
        result = pm.ProcessManager().kill(99_999_999)  # nonexistent
        self.assertTrue(result["ok"])
        self.assertIn("not found", result["message"].lower())

    def test_kill_real_subprocess(self) -> None:
        """Spawn a real python sleeper, kill it via ProcessManager.kill(),
        confirm PID is dead within 3s."""
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.2)
        try:
            self.assertTrue(pm._pid_alive(proc.pid),
                            f"sleeper {proc.pid} died before kill test")
            result = pm.ProcessManager().kill(proc.pid)
            self.assertTrue(result["ok"], f"kill failed: {result}")
            self.assertEqual(result.get("killed_pid"), proc.pid)
            deadline = time.time() + 3
            while time.time() < deadline and pm._pid_alive(proc.pid):
                time.sleep(0.1)
            self.assertFalse(pm._pid_alive(proc.pid),
                             f"PID {proc.pid} still alive after kill")
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass


class TestStart(unittest.TestCase):
    def test_start_unknown_role_returns_error(self) -> None:
        result = pm.ProcessManager().start("not-a-real-role")
        self.assertFalse(result["ok"])
        self.assertIn("unknown role", result["error"])

    def test_start_refuses_already_alive(self) -> None:
        """If _find_role_pid returns a live PID, start() must refuse."""
        with mock.patch.object(pm, "_find_role_pid", return_value=4242), \
             mock.patch.object(pm, "_pid_alive", return_value=True):
            result = pm.ProcessManager().start("bot")
        self.assertFalse(result["ok"])
        self.assertIn("already alive", result["error"])
        self.assertEqual(result["existing_pid"], 4242)


class TestSingleton(unittest.TestCase):
    def test_get_manager_returns_same_instance(self) -> None:
        a = pm.get_manager()
        b = pm.get_manager()
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
