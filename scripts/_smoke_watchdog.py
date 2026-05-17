"""60-second auto-restart watchdog for the TFT cluster smoke.

What it does every 60s:
  1. Verify orchestrator (:7700) is up. If not, kill stragglers + relaunch with --host 0.0.0.0.
  2. Verify local Razer worker (:7701) is up. If not, kill stragglers + relaunch.
  3. Verify the current smoke task is making progress:
     - status == "done" -> emit FINAL_SUCCESS and exit 0
     - status == "failed"/"cancelled" -> resubmit a new task, continue watching
     - status == "running":
         * if heartbeat (cluster task_update) > 5 min stale -> declare hung, kill worker, resubmit
         * else: emit progress line and keep watching
  4. Tail logs for new ERROR/Traceback lines, surface them.

The script's stdout is the event stream — every line becomes a notification
in the Monitor harness. ASCII-only output (per the new global rule).

Use:
  python scripts/_smoke_watchdog.py --task <task_id>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("d:/test 2/AI trading assistance")
LOG_DIR = ROOT / "logs"
VENV_PY = ROOT / "venv/Scripts/python.exe"
ORCH_URL = "http://127.0.0.1:7700"
WORKER_URL = "http://127.0.0.1:7701"
API_KEY = "AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9"

DEAD_HEARTBEAT_SEC = 600           # 10 min without task_update = hung
LOG_TAIL_LINES = 200
RESUBMIT_COOLDOWN_S = 60           # don't resubmit faster than this


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def http_get(url: str, timeout: float = 5.0):
    try:
        return requests.get(url, headers={"X-API-Key": API_KEY}, timeout=timeout)
    except Exception:
        return None


def http_post(url: str, body: dict, timeout: float = 10.0):
    try:
        return requests.post(
            url,
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=timeout,
        )
    except Exception:
        return None


def start_detached(cmd: str, log_file: Path | None = None) -> int | None:
    """Win32_Process.Create via powershell so the child has no parent in our shell tree."""
    if log_file is not None:
        inner = f'{cmd} >> "{log_file}" 2>&1'
        cmd = f'cmd /S /C "{inner}"'
    ps = (
        '$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments '
        '@{ CommandLine = $env:_CMD_; CurrentDirectory = $env:_CWD_ } -ErrorAction Stop; '
        'Write-Output $r.ProcessId'
    )
    env = os.environ.copy()
    env["_CMD_"] = cmd
    env["_CWD_"] = str(ROOT)
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            env=env, timeout=10,
        )
        pid_str = out.decode("utf-8", errors="replace").strip()
        return int(pid_str) if pid_str.isdigit() else None
    except Exception as e:
        print(f"[{now_ts()}] start_detached_err: {e}")
        return None


def kill_processes_matching(pattern: str) -> int:
    ps = (
        f'Get-WmiObject Win32_Process -Filter "Name=\'python.exe\'" 2>$null | '
        f'Where-Object {{ $_.CommandLine -match \'{pattern}\' }} | '
        'ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }}'
    )
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps], timeout=15,
        )
        killed = [l for l in out.decode("utf-8", errors="replace").splitlines() if l.strip()]
        return len(killed)
    except Exception:
        return 0


def ensure_orchestrator() -> bool:
    r = http_get(f"{ORCH_URL}/api/cluster/status", timeout=3)
    if r is not None and r.status_code == 200:
        return True
    print(f"[{now_ts()}] ORCH_DOWN -> restarting...")
    kill_processes_matching("distributed\\.orchestrator")
    time.sleep(2)
    pid = start_detached(
        f'"{VENV_PY}" -m src.training.distributed.orchestrator --port 7700 --host 0.0.0.0',
        LOG_DIR / "cluster.log",
    )
    print(f"[{now_ts()}] ORCH_STARTED pid={pid}")
    # Wait up to 30s for it to come up
    for _ in range(15):
        time.sleep(2)
        r = http_get(f"{ORCH_URL}/api/cluster/status", timeout=3)
        if r is not None and r.status_code == 200:
            print(f"[{now_ts()}] ORCH_UP_OK")
            return True
    print(f"[{now_ts()}] ORCH_UP_FAIL")
    return False


def ensure_worker() -> bool:
    """Check :7701. If down, restart the local Razer worker."""
    try:
        r = requests.get(f"{WORKER_URL}/health", timeout=3)
        if r.status_code in (200, 404):
            return True
    except Exception:
        pass
    # Some workers don't expose /health -- check the process listing instead
    ps = (
        'Get-WmiObject Win32_Process -Filter "Name=\'python.exe\'" 2>$null | '
        'Where-Object { $_.CommandLine -match \'distributed\\.worker\' } | '
        'Measure-Object | ForEach-Object { $_.Count }'
    )
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps], timeout=10,
        ).decode("utf-8", errors="replace").strip()
        if out and out != "0":
            return True
    except Exception:
        pass
    print(f"[{now_ts()}] WORKER_DOWN -> restarting...")
    kill_processes_matching("distributed\\.worker")
    time.sleep(2)
    pid = start_detached(
        f'"{VENV_PY}" -m src.training.distributed.worker '
        f'--master http://127.0.0.1:7700 --name RAZER --lane gpu --host 127.0.0.1',
        LOG_DIR / "worker_razer.log",
    )
    print(f"[{now_ts()}] WORKER_STARTED pid={pid}")
    # Give it 30s to register
    for _ in range(15):
        time.sleep(2)
        r = http_get(f"{ORCH_URL}/api/cluster/status", timeout=3)
        if r is not None and r.status_code == 200:
            try:
                workers = (r.json() or {}).get("workers") or []
                if any(w.get("status") in ("idle", "busy") for w in workers):
                    print(f"[{now_ts()}] WORKER_REGISTERED")
                    return True
            except Exception:
                pass
    print(f"[{now_ts()}] WORKER_REGISTER_TIMEOUT")
    return False


def get_task(task_id: str) -> dict | None:
    r = http_get(f"{ORCH_URL}/api/cluster/status", timeout=5)
    if r is None or r.status_code != 200:
        return None
    try:
        payload = r.json() or {}
    except Exception:
        return None
    for t in payload.get("recent_tasks") or []:
        if t.get("task_id") == task_id:
            return t
    return None


def parse_iso(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def submit_smoke() -> str | None:
    body = {
        "model_type": "tft",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "config": {
            "n_epochs": 1,
            "min_epochs": 1,
            "patience": 2,
            "input_chunk_length": 168,
            "output_chunk_length": 24,
            "history_days": 30,
            "sweep_id": f"smoke_auto_{int(time.time())}",
            "use_master_trainer": True,
        },
    }
    r = http_post(f"{ORCH_URL}/api/cluster/submit", body, timeout=10)
    if r is None or r.status_code != 200:
        print(f"[{now_ts()}] SUBMIT_FAIL status={r.status_code if r else 'no_resp'}")
        return None
    try:
        return (r.json() or {}).get("task_id")
    except Exception:
        return None


def get_progress() -> dict | None:
    try:
        p = json.loads((ROOT / "data/training_progress.json").read_text())
    except Exception:
        return None
    tasks = p.get("tasks") or {}
    tft = [(k, v) for k, v in tasks.items() if v.get("model") == "tft" and v.get("status") in ("running", "done")]
    if not tft:
        return None
    # Latest started
    tft.sort(key=lambda x: x[1].get("started_at", 0))
    return tft[-1][1]


def gpu_sample() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode("utf-8", errors="replace").splitlines()[0].strip().replace(" ", "")
        return out
    except Exception:
        return "no_gpu"


def tail_errors() -> str:
    """Last 2 new ERROR/Traceback lines across cluster + worker logs."""
    pat = ("ERROR", "Traceback", "FAILED", "UnicodeEncodeError")
    found = []
    for name in ("cluster.log", "worker_razer.log", "training.log"):
        p = LOG_DIR / name
        if not p.exists():
            continue
        try:
            lines = p.read_bytes().splitlines()[-LOG_TAIL_LINES:]
        except Exception:
            continue
        for ln in lines:
            try:
                t = ln.decode("utf-8", errors="replace")
            except Exception:
                continue
            if any(k in t for k in pat):
                found.append(f"{name}:{t[:160]}")
    return " || ".join(found[-2:]) if found else "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    task_id = args.task
    start_ts = time.time()
    iter_n = 0
    last_resubmit_ts = 0.0
    last_status = None
    last_progress_elapsed = 0.0
    stuck_iters = 0

    print(f"[{now_ts()}] WATCHDOG_START task={task_id}")

    while True:
        iter_n += 1
        elapsed_total = int(time.time() - start_ts)

        orch_ok = ensure_orchestrator()
        worker_ok = ensure_worker() if orch_ok else False
        task = get_task(task_id) if orch_ok else None
        status = (task or {}).get("status") or "unknown"
        assigned = (task or {}).get("assigned_to") or "-"
        prog = get_progress() or {}
        prog_status = prog.get("status", "?")
        prog_epoch = f"{prog.get('current_epoch','?')}/{prog.get('n_epochs','?')}"
        prog_elapsed = float(prog.get("elapsed_s") or 0)
        gpu = gpu_sample()
        errs = tail_errors()

        # Terminal handling
        if status == "done":
            print(f"[{now_ts()}] T+{elapsed_total}s iter={iter_n} task=DONE assigned={assigned} progress={prog_status} epoch={prog_epoch} elapsed={prog_elapsed:.0f}s gpu={gpu}")
            print(f"[{now_ts()}] FINAL_SUCCESS task={task_id}")
            return 0

        if status in ("failed", "cancelled"):
            now = time.time()
            if now - last_resubmit_ts < RESUBMIT_COOLDOWN_S:
                print(f"[{now_ts()}] STATUS={status} but RESUBMIT_COOLDOWN ({int(now-last_resubmit_ts)}s)")
            else:
                err = (task or {}).get("error") or "?"
                print(f"[{now_ts()}] STATUS={status} err={err[:140]} -> RESUBMITTING")
                new_id = submit_smoke()
                if new_id:
                    print(f"[{now_ts()}] RESUBMITTED new_task={new_id}")
                    task_id = new_id
                    last_resubmit_ts = now
                    stuck_iters = 0
                else:
                    print(f"[{now_ts()}] RESUBMIT_FAIL")
            time.sleep(60)
            continue

        # Stuck detection: use CLUSTER task last_update_at (heartbeats every 60s
        # from worker to orch). training_progress.elapsed_s only updates per-epoch
        # so it stays 0 during multi-symbol feature engineering -- not a reliable
        # liveness signal. Cluster heartbeats are the actual lifeline.
        last_update_ts = parse_iso((task or {}).get("last_update_at") or "")
        seconds_since_heartbeat = (time.time() - last_update_ts) if last_update_ts else 0
        if status == "running" and seconds_since_heartbeat > 300:
            # >5 min without a cluster task_update = worker hung
            stuck_iters += 1
        else:
            stuck_iters = 0

        if stuck_iters >= 5:  # 5 ticks (~5 min) without heartbeat = kill+resubmit
            print(f"[{now_ts()}] STUCK ({stuck_iters} iters without progress) -> killing worker + resubmitting")
            kill_processes_matching("distributed\\.worker")
            time.sleep(3)
            ensure_worker()
            now = time.time()
            if now - last_resubmit_ts >= RESUBMIT_COOLDOWN_S:
                new_id = submit_smoke()
                if new_id:
                    task_id = new_id
                    last_resubmit_ts = now
                    stuck_iters = 0
                    print(f"[{now_ts()}] STUCK_RESUBMIT new_task={task_id}")

        print(f"[{now_ts()}] T+{elapsed_total}s iter={iter_n} task={status} assigned={assigned[:18]} progress={prog_status} epoch={prog_epoch} elapsed={prog_elapsed:.0f}s gpu={gpu} errs={errs[:120]}")
        last_status = status
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
