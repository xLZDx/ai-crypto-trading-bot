"""Stage-1 cluster sweep watchdog.

Polls the orch every 60s, emits a one-line summary every iter. Surfaces:
  * task counts by status (pending / running / done / failed / blocked)
  * worker activity (which worker, current task, GPU%, CPU%)
  * any new ERROR / Traceback / FAILED in cluster + worker logs
  * estimated remaining time = pending * avg_completed_duration

Exits when all non-blocked tasks are terminal (done/failed/cancelled).

Usage:
  python scripts/_stage1_summary_watchdog.py --sweep-id full_retrain_2026-05-16
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("d:/test 2/AI trading assistance")
ORCH = "http://127.0.0.1:7700"
KEY  = "AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9"
H    = {"X-API-Key": KEY, "Content-Type": "application/json"}


def fetch_status():
    try:
        return requests.get(f"{ORCH}/api/cluster/status", headers=H, timeout=10).json()
    except Exception as e:
        return {"err": str(e)}


def fetch_state():
    try:
        return json.loads((ROOT / "data/orchestrator_state.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail_errors() -> str:
    pats = ("ERROR", "Traceback", "FAILED", "UnicodeEncodeError", "OutOfMemoryError")
    last = []
    for fname in ("cluster.log", "worker_razer.log", "training.log"):
        p = ROOT / "logs" / fname
        if not p.exists():
            continue
        try:
            lines = p.read_bytes().splitlines()[-150:]
        except Exception:
            continue
        for ln in lines:
            t = ln.decode("utf-8", errors="replace")
            if any(k in t for k in pats):
                last.append(f"{fname}:{t[-160:]}")
    return " | ".join(last[-2:]) if last else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", default="full_retrain_2026-05-16",
                    help="Substring match on sweep_id (so both full_retrain_2026-05-16 and full_retrain_2026-05-16_per_symbol count)")
    ap.add_argument("--max-iters", type=int, default=240)  # 240*60s = 4h cap per Monitor budget
    args = ap.parse_args()

    iter_n = 0
    start = time.time()
    last_done = 0
    completed_durations: list[float] = []  # for ETA
    while iter_n < args.max_iters:
        iter_n += 1
        elapsed_total = int(time.time() - start)
        st = fetch_state()
        tasks = st.get("tasks") or {}
        # Sweep filter
        sweep_tasks = [t for t in tasks.values()
                        if (t.get("config") or {}).get("sweep_id") == args.sweep_id]
        by_status = Counter(t.get("status") for t in sweep_tasks)
        by_model_status = Counter((t.get("model_type"), t.get("status")) for t in sweep_tasks)
        running = [t for t in sweep_tasks if t.get("status") == "running"]

        # Done in this iter? collect durations for ETA
        for t in sweep_tasks:
            if t.get("status") == "done":
                try:
                    s_ts = datetime.fromisoformat(t["started_at"].replace("Z","+00:00")).timestamp()
                    f_ts = datetime.fromisoformat(t["finished_at"].replace("Z","+00:00")).timestamp()
                    completed_durations.append(f_ts - s_ts)
                except Exception:
                    pass
        # Cap completed_durations memory
        if len(completed_durations) > 200:
            completed_durations = completed_durations[-200:]

        avg_dur = sum(completed_durations[-20:]) / max(1, len(completed_durations[-20:])) if completed_durations else 0
        pending_n = by_status.get("pending", 0)
        eta_min = (pending_n * avg_dur / 60) if avg_dur else 0

        # Worker activity
        wstat = fetch_status()
        wlines = []
        for w in (wstat.get("workers") or []):
            if not w.get("online"):
                continue
            wlines.append(f"{w.get('name')[:14]}/{w.get('lane')}={w.get('status')} ct={(w.get('current_task') or '-')[:8]} cpu={w.get('cpu_percent',0):.0f}% gpu={w.get('gpu_percent',0):.0f}%")
        worker_summary = " | ".join(wlines)

        errs = tail_errors()
        # Single-line summary
        status_str = " ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        print(f"[{datetime.now().strftime('%H:%M:%S')} +{elapsed_total}s iter={iter_n}] {status_str} | running_now={len(running)} avg_done_dur={avg_dur:.0f}s pending_eta_min={eta_min:.0f} | workers: {worker_summary} | errs: {errs[:120]}")

        # All-terminal? (no pending and no running)
        if by_status.get("pending", 0) == 0 and by_status.get("running", 0) == 0 and sum(by_status.values()) > 0:
            print("[ALL_TERMINAL] sweep complete")
            print(f"by_status_final={dict(by_status)}")
            print(f"by_model_status={dict(by_model_status)}")
            return 0
        time.sleep(60)
    print("[WATCHDOG_TIMEOUT] still running, will need re-launch")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
