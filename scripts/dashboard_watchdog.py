"""
dashboard_watchdog.py — keep the dashboard alive.

Polls http://127.0.0.1:5000/api/state every POLL_SECONDS. If the response is
non-200 (or socket times out) for FAILURE_THRESHOLD consecutive polls, kills
any stale dashboard process and respawns it via Win32_Process.Create
(Windows) or detached Popen (Unix) — same pattern restart_all.ps1 uses, so
the new dashboard survives this watchdog dying too.

Circuit breaker: if RESTART_LIMIT restarts happen within RESTART_WINDOW_S,
the watchdog stops trying and logs an alert. Operator must clear
data/dashboard_watchdog_state.json (or restart the watchdog) to resume —
prevents an infinite restart loop when the dashboard has a deterministic
import-time crash bug.

Surface (all under D: as project policy requires):
  - data/dashboard_watchdog_state.json — restart history + tripped flag
  - logs/dashboard_watchdog.log         — append-only event log

Independent of bot / dashboard / debug_supervisor so it survives THEIR
crashes — that's the whole point. Strictly project-scoped: only kills
processes whose command line matches `src.dashboard.app`.

Started by restart_all.ps1 alongside the dashboard.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH    = PROJECT_ROOT / 'logs' / 'dashboard_watchdog.log'
STATE_PATH  = PROJECT_ROOT / 'data' / 'dashboard_watchdog_state.json'
LAUNCH_PS1  = PROJECT_ROOT / 'launch_dashboard.ps1'
DASH_LOG    = PROJECT_ROOT / 'logs' / 'dashboard.log'

# Tunables — overridable via env vars so the operator can throttle without
# editing source.
POLL_SECONDS       = float(os.getenv('AI_TRADER_DASH_WATCH_POLL_S',   '10'))
HEALTH_TIMEOUT_S   = float(os.getenv('AI_TRADER_DASH_WATCH_TIMEOUT_S', '5'))
FAILURE_THRESHOLD  = int(  os.getenv('AI_TRADER_DASH_WATCH_FAIL_N',    '3'))
RESTART_LIMIT      = int(  os.getenv('AI_TRADER_DASH_WATCH_LIMIT',     '5'))
RESTART_WINDOW_S   = int(  os.getenv('AI_TRADER_DASH_WATCH_WINDOW_S', '600'))
HEALTH_URL         = os.getenv('AI_TRADER_DASH_WATCH_URL',
                               'http://127.0.0.1:5000/api/state')


def _log(msg: str, level: int = logging.INFO) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = (f'[{datetime.now(timezone.utc).isoformat()}] '
            f'{logging.getLevelName(level)} {msg}\n')
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass
    # Also echo to stdout so `tail -f logs/dashboard_watchdog.log` and
    # the launching console show the same thing.
    print(line.rstrip())


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {'restart_history': [], 'tripped': False}
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'restart_history': [], 'tripped': False}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Atomic write: temp + rename so a partial write can't corrupt
        # the next read.
        tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + '.tmp')
        tmp.write_text(json.dumps(state, indent=2, default=str),
                       encoding='utf-8')
        os.replace(tmp, STATE_PATH)
    except Exception as exc:
        _log(f'state write failed: {exc}', logging.WARNING)


def _check_health() -> bool:
    """True iff the dashboard responds 200 within HEALTH_TIMEOUT_S."""
    try:
        req = urllib.request.Request(HEALTH_URL,
                                     headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as resp:
            return resp.status == 200
    except (urllib.error.URLError, socket.timeout,
            ConnectionRefusedError, TimeoutError, OSError):
        return False
    except Exception:
        return False


def _kill_existing_dashboards() -> int:
    """Kill any python.exe whose command line matches src.dashboard.app.
    Returns the count killed. We do this BEFORE spawning a fresh one so
    we don't end up with two processes competing for port 5000."""
    try:
        import psutil
    except ImportError:
        _log('psutil unavailable — skipping kill (rely on port-bind '
             'failure to surface in logs)', logging.WARNING)
        return 0
    killed = 0
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if not (p.info.get('name') or '').lower().startswith('python'):
                continue
            cmd = ' '.join(p.info.get('cmdline') or [])
            if ('src.dashboard.app' in cmd
                    or 'src/dashboard/app' in cmd
                    or 'src\\dashboard\\app' in cmd):
                p.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        # Brief sleep so the OS releases the listen socket on :5000
        # before we try to bind it from the fresh process.
        time.sleep(2)
    return killed


def _spawn_dashboard() -> int:
    """Spawn a fresh dashboard fully detached from this watchdog so the
    new process survives even if THIS watchdog dies. Returns the new
    PID, or 0 on failure.

    Uses direct Popen with DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    | CREATE_BREAKAWAY_FROM_JOB on Windows — same pattern as
    _spawn_training_subprocess in src/dashboard/app.py. The earlier
    PowerShell-quoted Win32_Process.Create approach had a quoting bug
    that produced literal `\"` in the inner cmd line, so the spawned
    process died immediately and the watchdog logged 5+ failed
    restarts before being killed.
    """
    DASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Find the venv python — fall back to sys.executable (which IS the
    # venv python when the watchdog itself was launched from venv).
    venv_py = PROJECT_ROOT / 'venv' / 'Scripts' / 'python.exe'
    py = str(venv_py) if venv_py.exists() else sys.executable
    cmd = [py, '-m', 'src.dashboard.app']
    try:
        log_fp = open(DASH_LOG, 'ab', buffering=0)
    except Exception as exc:
        _log(f'log open failed: {exc}', logging.ERROR)
        return 0
    kw: dict = {
        'cwd':       str(PROJECT_ROOT),
        'stdout':    log_fp,
        'stderr':    log_fp,
        'stdin':     subprocess.DEVNULL,
        'close_fds': True,
    }
    # Pass UTF-8 io and the bind config that launch_dashboard.ps1 sets.
    env = os.environ.copy()
    env.setdefault('PYTHONIOENCODING',     'utf-8')
    env.setdefault('DASHBOARD_BIND_HOST',  '0.0.0.0')
    env.setdefault('DASHBOARD_BIND_PORT',  '5000')
    kw['env'] = env
    if sys.platform == 'win32':
        # 0x00000008 DETACHED_PROCESS · 0x00000200 CREATE_NEW_PROCESS_GROUP
        # · 0x01000000 CREATE_BREAKAWAY_FROM_JOB. Together: no console
        # parent, no inherited job. Stop-Process -Force on this watchdog
        # does not propagate to the new dashboard.
        kw['creationflags'] = (
            getattr(subprocess, 'DETACHED_PROCESS',          0x08)
            | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200)
            | 0x01000000
        )
    else:
        kw['start_new_session'] = True
    try:
        p = subprocess.Popen(cmd, **kw)
        return p.pid
    except Exception as exc:
        _log(f'spawn failed: {type(exc).__name__}: {exc}', logging.ERROR)
        return 0


def _circuit_tripped(state: dict[str, Any]) -> bool:
    """True if RESTART_LIMIT or more restarts happened within the rolling
    RESTART_WINDOW_S. Stops the watchdog from melting the box when the
    dashboard has a deterministic import-time crash."""
    now = time.time()
    cutoff = now - RESTART_WINDOW_S
    recent = [r for r in state.get('restart_history', [])
              if (r.get('ts') or 0) > cutoff]
    return len(recent) >= RESTART_LIMIT


def main() -> int:
    _log(f'watchdog starting (poll={POLL_SECONDS}s, '
         f'fail_threshold={FAILURE_THRESHOLD}, '
         f'limit={RESTART_LIMIT}/{RESTART_WINDOW_S}s)')
    state = _load_state()
    consecutive_failures = 0

    while True:
        # Operator can clear a tripped breaker by deleting the state file.
        # Otherwise we sit here logging once a minute so the operator's
        # `tail -f` shows we're still alive but waiting for them.
        if state.get('tripped'):
            _log('circuit breaker tripped — refusing to restart. Clear '
                 'data/dashboard_watchdog_state.json (or restart the '
                 'watchdog) to resume.', logging.ERROR)
            time.sleep(60)
            state = _load_state()  # picks up file deletion
            continue

        ok = _check_health()
        state['last_check_ts'] = time.time()
        if ok:
            consecutive_failures = 0
            _save_state(state)
            time.sleep(POLL_SECONDS)
            continue

        consecutive_failures += 1
        _log(f'health check failed ({consecutive_failures}/{FAILURE_THRESHOLD})',
             logging.WARNING)
        if consecutive_failures < FAILURE_THRESHOLD:
            _save_state(state)
            time.sleep(POLL_SECONDS)
            continue

        # Restart path.
        if _circuit_tripped(state):
            state['tripped'] = True
            state['tripped_at'] = time.time()
            _save_state(state)
            _log(f'circuit breaker TRIPPED ({RESTART_LIMIT}+ restarts in '
                 f'{RESTART_WINDOW_S}s) — halting restart loop',
                 logging.CRITICAL)
            consecutive_failures = 0
            continue

        _log('restarting dashboard...', logging.WARNING)
        killed = _kill_existing_dashboards()
        new_pid = _spawn_dashboard()
        state.setdefault('restart_history', []).append({
            'ts':           time.time(),
            'reason':       f'{FAILURE_THRESHOLD} consecutive failed health checks',
            'killed_count': killed,
            'new_pid':      new_pid,
        })
        # Cap at last 50 to keep the file small.
        state['restart_history'] = state['restart_history'][-50:]
        state['last_restart_ts'] = time.time()
        _save_state(state)
        if new_pid:
            _log(f'dashboard respawned (pid={new_pid}, killed={killed})')
        else:
            _log('dashboard respawn returned no pid — next health '
                 'check will reveal whether it actually started',
                 logging.ERROR)
        consecutive_failures = 0
        # Give the new dashboard time to bind :5000 + run its module
        # init (heavy ML imports take 5–15 s) before checking again.
        time.sleep(20)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _log('watchdog stopped by operator (KeyboardInterrupt)')
        sys.exit(0)
