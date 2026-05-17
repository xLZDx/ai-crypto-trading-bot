"""Autonomous cluster health monitor + auto-fix.

Run at each wakeup tick. Checks all services, auto-fixes issues, returns a
structured report dict. Prints a one-line summary; prints full report when
full_report=True is passed as argv[1].

Auto-fixes performed without approval:
  - Restart dead orch (if state file readable)
  - Restart dead workers (RAZER-GPU, RAZER-CPU, WORKER-1-GPU, WORKER-1-CPU)
  - Cancel stale `running` tasks whose worker has been offline >3 min
  - Resubmit tasks that went to `failed` during the sweep (up to 5 per run)
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT  = Path("d:/test 2/AI trading assistance")
ORCH  = "http://127.0.0.1:7700"
KEY   = "AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9"
H     = {"X-API-Key": KEY, "Content-Type": "application/json"}
VENV  = ROOT / "venv" / "Scripts" / "python.exe"
LOG   = ROOT / "logs"
SWEEP = "full_retrain_2026-05-16"

WORKER_DEFS = [
    # (name, lane, port, master_url, log_file, extra_args)
    ("RAZER",      "gpu", 7701, "http://127.0.0.1:7700", "worker_razer.log",     []),
    ("RAZER-CPU",  "cpu", 7703, "http://127.0.0.1:7700", "worker_razer_cpu.log", []),
]
IVAN_IP   = "192.168.0.167"
IVAN_USER = "koros"          # SSH username on Ivan
IVAN_VENV = "C:/ai-worker/venv/Scripts/python.exe"
IVAN_SCRIPT = "C:/ai-worker/ivan_node_monitor.py"
IVAN_WORKERS = [
    ("WORKER-1-GPU", 7701),
    ("WORKER-1-CPU", 7702),
]

STATE_FILE = ROOT / "data" / "_health_monitor_state.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"tick": 0, "last_full_report_ts": 0, "fixes_applied": []}


def _save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")


def _get(url: str, timeout: int = 8):
    return requests.get(url, headers=H, timeout=timeout).json()


def _delete(url: str):
    return requests.delete(url, headers=H, timeout=8)


def _post(url: str, body: dict):
    return requests.post(url, headers=H, data=json.dumps(body), timeout=10)


def check_orch() -> bool:
    try:
        r = requests.get(f"{ORCH}/api/cluster/status", headers=H, timeout=6)
        return r.status_code == 200
    except Exception:
        return False


def restart_orch() -> str:
    cmd = f'"{VENV}" -m src.training.distributed.orchestrator --port 7700 --host 0.0.0.0'
    log = str(LOG / "cluster.log")
    r = subprocess.run(
        ["powershell", "-Command",
         f'$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create '
         f'-Arguments @{{CommandLine=\'cmd /S /C "{cmd} >> \\"{log}\\" 2>&1"\';'
         f'CurrentDirectory=\'{ROOT}\'}};$r.ReturnValue'],
        capture_output=True, text=True, timeout=15
    )
    return f"orch restart rc={r.stdout.strip()}"


def restart_local_worker(name: str, lane: str, port: int, log_file: str) -> str:
    cmd = (f'"{VENV}" -m src.training.distributed.worker '
           f'--master http://127.0.0.1:7700 --name {name} --lane {lane} --port {port}')
    log = str(LOG / log_file)
    r = subprocess.run(
        ["powershell", "-Command",
         f'$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create '
         f'-Arguments @{{CommandLine=\'cmd /S /C "{cmd} >> \\"{log}\\" 2>&1"\';'
         f'CurrentDirectory=\'{ROOT}\'}};$r.ReturnValue'],
        capture_output=True, text=True, timeout=15
    )
    return f"{name} restart rc={r.stdout.strip()}"


def restart_ivan_workers_ssh(names: list) -> str:
    """SSH to Ivan and trigger the RestartWorkers scheduled task.

    restart_workers.ps1 requires Z:\\ (SMB share) which is only available in
    Ivan's interactive Windows session. The RestartWorkers scheduled task runs
    in that session and handles the Z:\\ mount correctly.
    """
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
        f"{IVAN_USER}@100.88.71.74",
        "powershell -NoProfile -Command \"Start-ScheduledTask -TaskName RestartWorkers\"",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return f"SSH_RESTART {names} rc={r.returncode}"
    except Exception as e:
        return f"SSH_RESTART FAIL {names}: {e}"


def probe_ivan_ssh() -> dict:
    """SSH to Ivan and run ivan_node_monitor.py — returns parsed JSON or error dict."""
    for ip in ("192.168.0.167", "100.88.71.74"):
        try:
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=6",
                "-o", "BatchMode=yes",
                f"{IVAN_USER}@{ip}",
                f'"{IVAN_VENV}" "{IVAN_SCRIPT}"',
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                data = json.loads(r.stdout.strip())
                data["_via"] = ip
                return data
        except Exception:
            continue
    return {"error": "SSH unreachable", "gpus": [], "cpu_pct": None}


def fix_stale_running(orch_online: bool, cancel_attempts: dict | None = None) -> list[str]:
    """Cancel tasks stuck in running state when their worker is offline.

    cancel_attempts: mutable dict {task_id -> attempt_count} persisted in monitor
    state. Tasks with >= 3 failed cancel attempts are reported as 'zombie_skip'
    (one-time) so the log isn't spammed every tick while the orch watchdog counts
    down to its own timeout. Pass None to use the legacy behaviour (always retry).
    """
    ZOMBIE_THRESHOLD = 3
    fixes = []
    if not orch_online:
        return fixes
    try:
        status = _get(f"{ORCH}/api/cluster/status")
        online_ids = {w["node_id"] for w in (status.get("workers") or []) if w.get("online")}
        sys.path.insert(0, str(ROOT / "src"))
        from utils.safe_json import read_json as _rj
        state = _rj(str(ROOT / "data/orchestrator_state.json")) or {}
        tasks = state.get("tasks") or {}
        for t in tasks.values():
            if t.get("status") != "running":
                continue
            assigned = t.get("worker_node_id") or t.get("assigned_to") or ""
            if not (assigned and assigned not in online_ids):
                continue
            tid = t["task_id"]
            if cancel_attempts is not None:
                attempts = cancel_attempts.get(tid, 0)
                if attempts >= ZOMBIE_THRESHOLD:
                    # Suppress repeat noise; log once when we cross the threshold
                    if attempts == ZOMBIE_THRESHOLD:
                        fixes.append(f"zombie_skip {tid[:8]} {t.get('model_type')} {t.get('symbol')} {t.get('timeframe')} (watchdog pending)")
                        cancel_attempts[tid] = ZOMBIE_THRESHOLD + 1  # don't log again
                    continue
                cancel_attempts[tid] = attempts + 1
            r = _delete(f"{ORCH}/api/cluster/task/{tid}")
            fixes.append(f"stale_cancel {tid[:8]} {t.get('model_type')} {t.get('symbol')} {t.get('timeframe')}")
    except Exception as e:
        fixes.append(f"stale_check_err: {e}")
    return fixes


def resubmit_failed(orch_online: bool, already_resubmitted: set[str],
                    max_resubmit: int = 5) -> list[str]:
    """Resubmit recently-failed sweep tasks (up to max_resubmit per run)."""
    fixes = []
    if not orch_online:
        return fixes
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from utils.safe_json import read_json as _rj
        state = _rj(str(ROOT / "data/orchestrator_state.json")) or {}
        tasks = state.get("tasks") or {}
        failed = [t for t in tasks.values()
                  if t.get("status") == "failed"
                  and t.get("task_id") not in already_resubmitted
                  and (t.get("config") or {}).get("sweep_id", "").startswith("full_retrain_2026-05-16")]
        for t in failed[:max_resubmit]:
            cfg = t.get("config") or {}
            spec = {
                "model_type": t.get("model_type"),
                "symbol":     t.get("symbol"),
                "timeframe":  t.get("timeframe"),
                "config": {
                    "sweep_id":         cfg.get("sweep_id", SWEEP),
                    "worker_name_pins": cfg.get("worker_name_pins"),
                    "wizard_fix":       cfg.get("wizard_fix", "asym_tb_pt4_sl2"),
                },
                "data_path":   t.get("data_path", ""),
                "output_path": t.get("output_path", str(ROOT / "models")),
            }
            r = _post(f"{ORCH}/api/cluster/submit", spec)
            if r.status_code == 200:
                fixes.append(f"resubmit {t.get('model_type')} {t.get('symbol')} {t.get('timeframe')}")
            else:
                fixes.append(f"resubmit_fail {t['task_id'][:8]} status={r.status_code}")
    except Exception as e:
        fixes.append(f"resubmit_err: {e}")
    return fixes


def tail_errors() -> list[str]:
    pats = ("ERROR", "Traceback", "FAILED", "OOM", "OutOfMemory", "CUDA out")
    found = []
    for fname in ("cluster.log", "worker_razer.log", "worker_razer_cpu.log", "training.log"):
        p = LOG / fname
        if not p.exists():
            continue
        try:
            lines = p.read_bytes().splitlines()[-80:]
        except Exception:
            continue
        for ln in lines[-80:]:
            t = ln.decode("utf-8", errors="replace")
            if any(k in t for k in pats):
                found.append(f"{fname}:{t[-120:].strip()}")
    return found[-4:]


def task_summary(orch_online: bool, min_done: int = 0) -> dict:
    """Read sweep stats from the state file, retrying until total looks valid.

    The orch occasionally flushes a partial in-memory queue snapshot that has
    fewer tasks than the full state, or temporarily marks done tasks as pending
    mid-flush.  Retry up to MAX_ATTEMPTS times (2 s apart) rejecting any read
    where total < EXPECTED_MIN_TASKS OR done < min_done - DONE_DROP_TOLERANCE.
    """
    EXPECTED_MIN_TASKS = 600   # full sweep has 642; reject any snapshot smaller
    DONE_DROP_TOLERANCE = 5    # allow up to 5 fewer done than our high-water mark
    MAX_ATTEMPTS = 6           # up to 10 s total wait (5 x 2 s sleeps)
    if not orch_online:
        return {}
    def _read() -> dict:
        from collections import Counter
        state = json.loads((ROOT / "data/orchestrator_state.json").read_text(encoding="utf-8"))
        tasks = state.get("tasks") or {}
        sweep = [t for t in tasks.values()
                 if (t.get("config") or {}).get("sweep_id", "").startswith("full_retrain_2026-05-16")]
        by_st = Counter(t.get("status") for t in sweep)
        by_mt_st = Counter((t.get("model_type"), t.get("status")) for t in sweep)
        done  = by_st.get("done", 0)
        total = sum(by_st.values())
        return {"by_status": dict(by_st), "by_mt_status": dict(by_mt_st),
                "done": done, "total": total,
                "pct": round(100 * done / max(1, total), 1)}
    try:
        best = None
        for attempt in range(MAX_ATTEMPTS):
            result = _read()
            total_ok = result.get("total", 0) >= EXPECTED_MIN_TASKS
            done_ok  = result.get("done", 0) >= min_done - DONE_DROP_TOLERANCE
            if total_ok and done_ok:
                return result
            # Keep the best read so far (highest done count) as fallback
            if best is None or result.get("done", 0) > best.get("done", 0):
                best = result
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2)
        return best if best is not None else result  # best we got after all retries
    except Exception as e:
        return {"err": str(e)}


def worker_lines(orch_online: bool) -> list[str]:
    if not orch_online:
        return ["(orch down)"]
    try:
        status = _get(f"{ORCH}/api/cluster/status")
        lines = []
        for w in (status.get("workers") or []):
            if not w.get("online"):
                continue
            ct = (w.get("current_task") or "-")[:8]
            lines.append(f"{w['name']:<11} lane={w['lane']:<3} ct={ct} cpu={w.get('cpu_percent',0):.0f}% gpu={w.get('gpu_percent',0):.0f}%")
        return lines or ["(no online workers)"]
    except Exception as e:
        return [f"status_err: {e}"]


def main():
    full = len(sys.argv) > 1 and sys.argv[1] == "full"
    st   = _load_state()
    st["tick"] = st.get("tick", 0) + 1
    now_ts = time.time()
    fixes  = []

    # 1. Check orch
    orch_ok = check_orch()
    if not orch_ok:
        msg = restart_orch()
        fixes.append(f"ORCH_RESTART: {msg}")
        time.sleep(8)
        orch_ok = check_orch()

    # 2. Check local workers (Razer-GPU + Razer-CPU)
    for name, lane, port, _, log_file, _ in [
        ("RAZER",     "gpu", 7701, None, "worker_razer.log",     []),
        ("RAZER-CPU", "cpu", 7703, None, "worker_razer_cpu.log", []),
    ]:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", headers=H, timeout=4)
            alive = r.status_code == 200
        except Exception:
            alive = False
        if not alive:
            msg = restart_local_worker(name, lane, port, log_file)
            fixes.append(f"WORKER_RESTART: {msg}")
            time.sleep(5)

    # 3. Check Ivan workers — group by name, only restart if latest entry offline.
    # Orch keeps one entry per node_id (one per restart), so we must pick the
    # most-recently-seen entry per name to determine true online state.
    if orch_ok:
        try:
            status = _get(f"{ORCH}/api/cluster/status")
            # latest_by_name: name -> (last_seen_ts, online)
            latest_by_name: dict[str, tuple[float, bool]] = {}
            for w in (status.get("workers") or []):
                name = w.get("name", "")
                if name not in ("WORKER-1-GPU", "WORKER-1-CPU"):
                    continue
                try:
                    from datetime import datetime as _dt
                    ls = w.get("last_seen") or ""
                    ts = _dt.fromisoformat(ls.replace("Z", "+00:00")).timestamp() if ls else 0.0
                except Exception:
                    ts = 0.0
                prev_ts, _ = latest_by_name.get(name, (0.0, False))
                if ts >= prev_ts:
                    latest_by_name[name] = (ts, bool(w.get("online")))
            # Restart workers whose latest entry is offline — always via SSH.
            # Cooldown: 300s between restarts so duplicate-spawn loops can't form
            # when new workers take >1 tick to register with the master.
            offline_names = [n for n, (_, online) in latest_by_name.items() if not online]
            if offline_names:
                last_restart_ts = st.get("last_ivan_restart_ts", 0.0)
                cooldown_remaining = 300 - (time.time() - last_restart_ts)
                if cooldown_remaining <= 0:
                    msg = restart_ivan_workers_ssh(offline_names)
                    st["last_ivan_restart_ts"] = time.time()
                    fixes.append(f"IVAN_SSH_RESTART: {msg}")
                else:
                    fixes.append(f"ivan_restart_skipped: cooldown {int(cooldown_remaining)}s")
        except Exception as e:
            fixes.append(f"ivan_check_err: {e}")

    # 4. Fix stale running tasks (track attempts to suppress repeat noise)
    cancel_attempts = st.setdefault("cancel_attempts", {})
    fixes += fix_stale_running(orch_ok, cancel_attempts)

    # 5. Resubmit failed sweep tasks (skip already-resubmitted IDs)
    already_resubmitted = set(st.get("resubmitted_ids") or [])
    new_fixes = resubmit_failed(orch_ok, already_resubmitted)
    # Track which task_ids we just resubmitted so we don't loop
    if new_fixes:
        state_disk = json.loads((ROOT / "data/orchestrator_state.json").read_text(encoding="utf-8"))
        for t in (state_disk.get("tasks") or {}).values():
            if (t.get("status") == "failed"
                    and t.get("task_id") not in already_resubmitted
                    and (t.get("config") or {}).get("sweep_id", "").startswith("full_retrain_2026-05-16")):
                already_resubmitted.add(t["task_id"])
        st["resubmitted_ids"] = list(already_resubmitted)[-200:]
    fixes += new_fixes

    # 6. Probe Ivan via SSH (non-blocking — skip if SSH not yet available)
    ivan_ssh = probe_ivan_ssh()

    # 7. Build report — pass high-water mark so stale mid-flush snapshots are rejected
    min_done = st.get("max_done_seen", 0)
    summary  = task_summary(orch_ok, min_done=min_done)
    # Update high-water mark
    if summary.get("done", 0) > min_done:
        st["max_done_seen"] = summary["done"]
    w_lines  = worker_lines(orch_ok)
    errs     = tail_errors()
    ts_str   = datetime.now().strftime("%H:%M:%S")
    done_pct = f"{summary.get('done',0)}/{summary.get('total','?')} ({summary.get('pct',0)}%)"

    # Format Ivan SSH line + RAM alert
    ivan_ok = "error" not in ivan_ssh or ivan_ssh.get("error") is None
    if ivan_ok:
        gpus = ivan_ssh.get("gpu", {}).get("gpus") or []
        gpu0 = gpus[0] if gpus else {}
        cpu_mem = ivan_ssh.get("cpu_mem", {})
        ram_pct = cpu_mem.get("ram_pct", 0) or 0
        ram_used = cpu_mem.get("ram_used_gb", 0) or 0
        ram_total = cpu_mem.get("ram_total_gb", 1) or 1
        ram_flag = " RAM_HIGH" if ram_pct >= 84 else ""
        ivan_line = (f"Ivan(SSH) cpu={cpu_mem.get('cpu_pct',0):.0f}%"
                     f" gpu={gpu0.get('gpu_pct',0):.0f}%"
                     f" vram={gpu0.get('mem_used_mb',0):.0f}/{gpu0.get('mem_total_mb',0):.0f}MB"
                     f" ram={ram_used:.1f}/{ram_total:.1f}GB({ram_pct:.0f}%)"
                     f" temp={gpu0.get('temp_c',0):.0f}C"
                     f" via={ivan_ssh.get('_via','?')}{ram_flag}")
        if ram_pct >= 90:
            fixes.append(f"IVAN_RAM_CRITICAL {ram_pct:.0f}% used — {ram_total-ram_used:.1f}GB free")
        elif ram_pct >= 85:
            fixes.append(f"IVAN_RAM_HIGH {ram_pct:.0f}% used — {ram_total-ram_used:.1f}GB free")
    else:
        ivan_line = f"Ivan(SSH) UNREACHABLE — {ivan_ssh.get('error','?')}"

    # One-liner always
    fix_str = f" | FIXES={fixes}" if fixes else ""
    print(f"[HEALTH {ts_str} tick={st['tick']}] orch={'UP' if orch_ok else 'DOWN'} tasks={done_pct}{fix_str}")
    for wl in w_lines:
        print(f"  {wl}")
    print(f"  {ivan_line}")
    if errs:
        print(f"  ERRORS: {' | '.join(errs[:2])}")

    # Full report every 10 min
    since_last = now_ts - st.get("last_full_report_ts", 0)
    if full or since_last >= 590:
        st["last_full_report_ts"] = now_ts
        print("\n" + "="*60)
        print(f"FULL STATUS REPORT  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        by_st = summary.get("by_status", {})
        print(f"Sweep tasks: {json.dumps(by_st)}")
        print(f"Progress:    {done_pct}")
        by_mt = summary.get("by_mt_status", {})
        done_by_mt: dict = {}
        pend_by_mt: dict = {}
        for (mt, s2), c in by_mt.items():
            if s2 == "done":   done_by_mt[mt] = c
            if s2 == "pending": pend_by_mt[mt] = c
        for mt in sorted(set(list(done_by_mt) + list(pend_by_mt))):
            print(f"  {mt:<14} done={done_by_mt.get(mt,0):>3}  pend={pend_by_mt.get(mt,0):>3}")
        print("Workers:")
        for wl in w_lines:
            print(f"  {wl}")
        print(f"Ivan node (SSH):")
        if ivan_ok:
            print(f"  {ivan_line}")
            gpus = ivan_ssh.get("gpu", {}).get("gpus") or []
            for i, g in enumerate(gpus):
                print(f"  GPU{i} {g.get('name','')} "
                      f"util={g.get('gpu_pct',0):.0f}% "
                      f"mem={g.get('mem_used_mb',0):.0f}/{g.get('mem_total_mb',0):.0f}MB "
                      f"temp={g.get('temp_c',0):.0f}C "
                      f"power={g.get('power_w',0):.0f}W")
            procs = ivan_ssh.get("procs") or []
            for proc in procs[:4]:
                print(f"  proc pid={proc.get('pid')} cpu={proc.get('cpu_pct',0)}% "
                      f"mem={proc.get('mem_mb',0):.0f}MB  {proc.get('cmd','')[-80:]}")
        else:
            print(f"  {ivan_line}")
        if fixes:
            print(f"Auto-fixes this tick: {fixes}")
        if errs:
            print("Log errors (last 2):")
            for e in errs[:2]:
                print(f"  {e[:180]}")
        print("="*60)

    st["fixes_applied"] = (st.get("fixes_applied") or [])[-50:]
    if fixes:
        st["fixes_applied"].extend(fixes)
    _save_state(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
