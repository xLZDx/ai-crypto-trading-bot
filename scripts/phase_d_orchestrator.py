"""phase_d_orchestrator — fires after master's hybrid sweep completes,
splits the remaining heavy GPU work (TFT × 2 TFs + OFT × 3 symbols)
across master + Ivan, waits for all tasks, then triggers the final
multi-TF backtest with everything fresh.

Designed to hit the 10:00 Chișinău (07:00 UTC) deadline by:
  1. Polling pipeline_status.json until status=='done' (master sweep
     finished its meta + chained backtest) — frees master CPU/GPU.
  2. Spawning a worker process on master itself that registers with
     the local cluster orchestrator on port 7700 as 'LOCAL_RAZER',
     so master becomes a peer-equal cluster node.
  3. Submitting 5 cluster tasks:
        TFT @ 1h, TFT @ 4h, OFT BTC, OFT ETH, OFT SOL
     The cluster's load balancer assigns idle workers; with 2 nodes
     the wall clock is roughly half of single-node.
  4. Polling task status until all finish.
  5. Re-running run_full_backtest() with no filters to produce the
     final all-fresh-models heatmap + comparison CSV.

State persists to data/audit_reports/phase_d_state.json so a
restart resumes where it left off.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH       = PROJECT_ROOT / 'logs' / 'phase_d_orchestrator.log'
STATE_PATH     = PROJECT_ROOT / 'data' / 'audit_reports' / 'phase_d_state.json'
PIPELINE_STATUS = PROJECT_ROOT / 'data' / 'pipeline_status.json'

CLUSTER_BASE = 'http://192.168.0.105:7700'
SMB_BASE     = 'Z:'    # worker-side mount; master uses its own paths

POLL_S = 30


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{datetime.now(timezone.utc).isoformat()}] {msg}\n'
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line)
    print(line, end='', file=sys.stderr)


def _save_state(s: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, STATE_PATH)


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'phase': 'wait_for_master_sweep', 'started_at': time.time(),
            'task_ids': [], 'master_worker_pid': None, 'transitions': []}


def _http_get(url: str, timeout: int = 5) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        _log(f'GET {url} failed: {e}')
        return None


def _http_post(url: str, payload: dict, timeout: int = 10) -> dict | None:
    try:
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=body, method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        _log(f'POST {url} failed: {e}')
        return None


def _master_sweep_done() -> bool:
    if not PIPELINE_STATUS.exists():
        return False
    try:
        s = json.loads(PIPELINE_STATUS.read_text(encoding='utf-8'))
        return (s.get('status') == 'done'
                and s.get('train', {}).get('ok') is True
                and s.get('backtest') is not None)
    except Exception:
        return False


def _spawn_master_as_worker() -> int | None:
    """Spawn a worker.py process on master itself. Master becomes a
    cluster node named LOCAL_RAZER."""
    venv = PROJECT_ROOT / 'venv' / 'Scripts' / 'python.exe'
    log_out = PROJECT_ROOT / 'logs' / 'master_worker.log'
    log_err = PROJECT_ROOT / 'logs' / 'master_worker.err.log'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    cmd = [str(venv), '-m', 'src.training.distributed.worker',
           '--master', CLUSTER_BASE, '--name', 'LOCAL_RAZER']
    flags = 0
    if sys.platform == 'win32':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=open(log_out, 'a', encoding='utf-8'),
        stderr=open(log_err, 'a', encoding='utf-8'),
        creationflags=flags,
    )
    _log(f'Spawned LOCAL_RAZER worker on master, PID {proc.pid}')
    return proc.pid


def _submit_phase_d_tasks() -> list[str]:
    """Submit 5 GPU-heavy tasks to the cluster.

    Path strategy: workers see master's project at Z:\\ (SMB mount);
    master itself sees it as the project root. So we send a UNC-style
    data_path that works for the worker, and trust that master's worker
    handler accepts it via the share-equivalent local path."""
    tasks = [
        # TFT — neural attention forecast. Worker GPU 6GB is plenty.
        {'model_type': 'tft', 'timeframe': '1h', 'symbol': 'ALL',
         'data_path': '', 'output_path': '',
         'config': {'use_master_trainer': True}},
        {'model_type': 'tft', 'timeframe': '4h', 'symbol': 'ALL',
         'data_path': '', 'output_path': '',
         'config': {'use_master_trainer': True}},
        # OFT — microstructure, per-symbol. Split across nodes.
        {'model_type': 'oft', 'timeframe': '1m', 'symbol': 'BTC/USDT',
         'data_path': '', 'output_path': '',
         'config': {'use_master_trainer': True}},
        {'model_type': 'oft', 'timeframe': '1m', 'symbol': 'ETH/USDT',
         'data_path': '', 'output_path': '',
         'config': {'use_master_trainer': True}},
        {'model_type': 'oft', 'timeframe': '1m', 'symbol': 'SOL/USDT',
         'data_path': '', 'output_path': '',
         'config': {'use_master_trainer': True}},
    ]
    ids: list[str] = []
    for t in tasks:
        r = _http_post(f'{CLUSTER_BASE}/api/cluster/submit', t)
        if r and r.get('ok'):
            ids.append(r['task_id'])
            _log(f'Submitted {t["model_type"]} @ {t["timeframe"]} ({t["symbol"]}) → task {r["task_id"]}')
        else:
            _log(f'Submit FAILED for {t}: {r}')
    return ids


def _wait_for_tasks(task_ids: list[str], poll_s: int = 60) -> dict:
    """Poll until all task_ids reach a terminal status (done/failed/cancelled).
    Returns dict {task_id: status}."""
    seen_done: dict[str, str] = {}
    while len(seen_done) < len(task_ids):
        all_tasks = _http_get(f'{CLUSTER_BASE}/api/cluster/tasks')
        if not all_tasks:
            time.sleep(poll_s)
            continue
        for t in all_tasks:
            tid = t.get('task_id')
            if tid in task_ids and tid not in seen_done:
                st = t.get('status', '?')
                if st in ('done', 'failed', 'cancelled'):
                    seen_done[tid] = st
                    elapsed = t.get('elapsed_s', 0)
                    err = t.get('error', '')
                    _log(f'Task {tid[:8]} {t.get("model_type","?")}@{t.get("timeframe","?")}({t.get("symbol","?")}) → {st} (elapsed={elapsed:.0f}s err={err[:80]})')
        if len(seen_done) < len(task_ids):
            time.sleep(poll_s)
    return seen_done


def _trigger_final_backtest() -> None:
    """Run run_full_backtest with no model filter — covers all 27 strategies
    × 5 TFs × 20 symbols against all-fresh models."""
    venv = PROJECT_ROOT / 'venv' / 'Scripts' / 'python.exe'
    log_out = PROJECT_ROOT / 'logs' / 'phase_d_final_backtest.log'
    cmd = [str(venv), '-c',
           "from src.engine.backtester import run_full_backtest; "
           "df = run_full_backtest(timeframes=('5m','15m','1h','4h','1d')); "
           "print(f'rows={len(df)}')"]
    _log(f'Triggering final backtest: {" ".join(cmd[:3])} ...')
    with open(log_out, 'a', encoding='utf-8') as f:
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    _log(f'Final backtest exit code: {rc}')


def main() -> int:
    state = _load_state()
    _log(f'phase_d_orchestrator starting; phase={state["phase"]}')

    # Phase 1 — wait for master's hybrid sweep to finish meta + backtest
    if state['phase'] == 'wait_for_master_sweep':
        _log('Polling pipeline_status.json for master sweep completion...')
        while not _master_sweep_done():
            time.sleep(POLL_S)
        _log('Master sweep complete — entering Phase D dispatch')
        state['phase'] = 'spawn_master_worker'
        state['transitions'].append({'ts': time.time(), 'to': 'spawn_master_worker'})
        _save_state(state)

    # Phase 2 — make master a cluster node
    if state['phase'] == 'spawn_master_worker':
        pid = _spawn_master_as_worker()
        state['master_worker_pid'] = pid
        time.sleep(15)   # let it register
        # Verify two workers online
        s = _http_get(f'{CLUSTER_BASE}/api/cluster/status')
        n = s.get('workers_online', 0) if s else 0
        _log(f'cluster shows {n} workers online (expecting 2: LOCAL_RAZER + Ivan)')
        state['phase'] = 'submit_tasks'
        state['transitions'].append({'ts': time.time(), 'to': 'submit_tasks',
                                      'workers_online': n})
        _save_state(state)

    # Phase 3 — submit Phase D tasks
    if state['phase'] == 'submit_tasks':
        ids = _submit_phase_d_tasks()
        if not ids:
            _log('No tasks submitted — aborting')
            return 1
        state['task_ids'] = ids
        state['phase'] = 'wait_for_tasks'
        state['transitions'].append({'ts': time.time(), 'to': 'wait_for_tasks',
                                      'submitted': len(ids)})
        _save_state(state)

    # Phase 4 — wait for all tasks
    if state['phase'] == 'wait_for_tasks':
        statuses = _wait_for_tasks(state['task_ids'])
        state['task_statuses'] = statuses
        state['phase'] = 'final_backtest'
        state['transitions'].append({'ts': time.time(), 'to': 'final_backtest',
                                      'statuses': statuses})
        _save_state(state)

    # Phase 5 — final backtest
    if state['phase'] == 'final_backtest':
        _trigger_final_backtest()
        state['phase'] = 'done'
        state['done_at'] = time.time()
        state['transitions'].append({'ts': time.time(), 'to': 'done'})
        _save_state(state)
        _log('Phase D orchestrator complete')

    return 0


if __name__ == '__main__':
    sys.exit(main())
