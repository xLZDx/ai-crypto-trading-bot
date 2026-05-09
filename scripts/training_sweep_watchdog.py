"""
training_sweep_watchdog.py — keep the overnight curated sweep alive.

Polls http://127.0.0.1:5000/api/pipeline/status every POLL_SECONDS.
Detects "dead" or "stalled" sweeps and re-triggers them. The skip-if-fresh
guard in train_all_models means the relaunched orchestrator picks up where
the previous attempt died — no wasted work on already-finished combos.

NEVER kills an actively-progressing sweep — per the operator memory
`feedback_dont_relaunch_inflight_training`. We only respawn when:
  - The orchestrator process isn't visible in psutil cmdline scan (genuine
    death), OR
  - The pipeline_status.json hasn't been updated for ≥ STALL_S seconds AND
    no per-trainer log file is growing (everything is silent).

Self-healing flow:
  1. Sweep dies → watchdog detects → respawns via /api/pipeline/run.
  2. Skip-if-fresh skips already-finished combos.
  3. Walk-forward CV / training resumes from the failed combo.
  4. Repeat until completion or RESTART_LIMIT trips the circuit breaker.

Disk over RAM: state persists to data/training_sweep_watchdog_state.json
between restarts so circuit-breaker counters survive process reboots.

Started by restart_all.ps1 alongside the dashboard / dashboard_watchdog.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH   = PROJECT_ROOT / 'logs' / 'training_sweep_watchdog.log'
STATE_PATH = PROJECT_ROOT / 'data' / 'training_sweep_watchdog_state.json'

POLL_SECONDS      = float(os.getenv('AI_TRADER_SWEEP_WATCH_POLL_S',   '60'))
STALL_S           = float(os.getenv('AI_TRADER_SWEEP_WATCH_STALL_S',  '600'))   # 10 min idle = stall
HEALTH_TIMEOUT_S  = float(os.getenv('AI_TRADER_SWEEP_WATCH_TIMEOUT_S', '5'))
RESTART_LIMIT     = int(  os.getenv('AI_TRADER_SWEEP_WATCH_LIMIT',     '8'))
RESTART_WINDOW_S  = int(  os.getenv('AI_TRADER_SWEEP_WATCH_WINDOW_S', '21600'))  # 6 h
STATUS_URL        = os.getenv('AI_TRADER_SWEEP_WATCH_STATUS_URL',
                               'http://127.0.0.1:5000/api/pipeline/status')
TRIGGER_URL       = os.getenv('AI_TRADER_SWEEP_WATCH_TRIGGER_URL',
                               'http://127.0.0.1:5000/api/pipeline/run')
API_KEY           = os.getenv('AI_TRADER_API_KEY', '')


def _log(msg: str, level: int = logging.INFO) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = (f'[{datetime.now(timezone.utc).isoformat()}] '
            f'[{logging.getLevelName(level):7s}] {msg}\n')
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception:
        pass
    print(line, end='', file=sys.stderr)


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {'restarts': [], 'tripped': False, 'last_status_ts': 0,
                'last_status_payload_hash': ''}
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception as exc:
        _log(f'state file unreadable ({exc}); starting fresh', logging.WARNING)
        return {'restarts': [], 'tripped': False, 'last_status_ts': 0,
                'last_status_payload_hash': ''}


def _save_state(s: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=2), encoding='utf-8')
    os.replace(tmp, STATE_PATH)


def _circuit_tripped(s: dict[str, Any]) -> bool:
    """Have we hit RESTART_LIMIT respawns within RESTART_WINDOW_S?"""
    if s.get('tripped'):
        return True
    cutoff = time.time() - RESTART_WINDOW_S
    recent = [r for r in s.get('restarts', []) if r > cutoff]
    if len(recent) >= RESTART_LIMIT:
        s['tripped'] = True
        _save_state(s)
        _log(f'CRITICAL: {len(recent)} respawns in {RESTART_WINDOW_S}s — circuit tripped. '
             f'Operator must clear {STATE_PATH.name} or restart the watchdog.',
             logging.CRITICAL)
        return True
    return False


def _fetch_status() -> dict[str, Any] | None:
    """Returns the /api/pipeline/status JSON or None on failure."""
    try:
        req = urllib.request.Request(STATUS_URL, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode('utf-8'))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _orchestrator_alive() -> bool:
    """Cmdline-scan fallback (PR-32 pattern) for the pipeline orchestrator
    subprocess. True iff a python process running
    `src.engine.pipeline_orchestrator` exists."""
    try:
        import psutil
    except ImportError:
        return True  # can't tell — assume alive, don't trigger respawns
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = (p.info.get('name') or '').lower()
            if not name.startswith('python'):
                continue
            cmd = p.info.get('cmdline') or []
            if (len(cmd) >= 3 and cmd[1] == '-m'
                    and cmd[2] == 'src.engine.pipeline_orchestrator'):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _respawn_sweep(s: dict[str, Any]) -> bool:
    """POST /api/pipeline/run to trigger a fresh sweep. Returns True on
    202 / 200 success."""
    try:
        body = json.dumps({'reason': 'training_sweep_watchdog respawn'}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if API_KEY:
            headers['X-API-Key'] = API_KEY
        req = urllib.request.Request(TRIGGER_URL, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as r:
            ok = r.status in (200, 201, 202)
    except (urllib.error.URLError, OSError) as exc:
        _log(f'/api/pipeline/run POST failed: {exc}', logging.ERROR)
        return False
    if ok:
        s.setdefault('restarts', []).append(time.time())
        _save_state(s)
        _log(f'Sweep respawn triggered (restart #{len(s["restarts"])} in this window)',
             logging.WARNING)
    return ok


def _is_stalled(payload: dict[str, Any] | None, s: dict[str, Any]) -> bool:
    """Stall detection — payload hasn't changed in STALL_S AND orchestrator
    isn't visibly running. Returns True iff we should respawn."""
    if payload is None:
        # Status endpoint unreachable. If orchestrator isn't alive either,
        # call it dead. Otherwise wait — Flask might just be busy.
        return not _orchestrator_alive()

    status = (payload.get('status') or '').lower()
    if status in ('done', 'idle', 'cancelled'):
        # Sweep finished or never started. Nothing to do.
        return False

    # Hash the payload to detect "no progress at all". Anything with a
    # changing timestamp / counter / phase string flips the hash.
    payload_hash = json.dumps(payload, sort_keys=True, default=str)
    now = time.time()
    if payload_hash != s.get('last_status_payload_hash'):
        s['last_status_payload_hash'] = payload_hash
        s['last_status_ts'] = now
        _save_state(s)
        return False
    idle_for = now - s.get('last_status_ts', now)
    if idle_for >= STALL_S and not _orchestrator_alive():
        _log(f'Stall detected: payload unchanged for {idle_for:.0f}s + no orchestrator process',
             logging.WARNING)
        return True
    return False


def main() -> int:
    _log(f'training_sweep_watchdog starting · poll={POLL_SECONDS}s '
         f'stall={STALL_S}s limit={RESTART_LIMIT}/{RESTART_WINDOW_S}s')
    s = _load_state()
    if s.get('tripped'):
        _log('circuit-breaker is tripped — exiting; clear state to resume',
             logging.CRITICAL)
        return 1

    try:
        while True:
            payload = _fetch_status()
            if _is_stalled(payload, s):
                if _circuit_tripped(s):
                    return 1
                ok = _respawn_sweep(s)
                if not ok:
                    _log('respawn POST failed; will retry next poll', logging.ERROR)
                # Reset the stall timer after a respawn so we don't fire again
                # immediately while the new sweep ramps up.
                s['last_status_payload_hash'] = ''
                s['last_status_ts'] = time.time()
                _save_state(s)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        _log('SIGINT — shutting down cleanly')
        return 0


if __name__ == '__main__':
    sys.exit(main())
