"""Live training-progress monitor for the 1-epoch TFT smoke run.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\monitor_smoke.py
    .\\venv\\Scripts\\python.exe scripts\\monitor_smoke.py --once    # single snapshot

Refreshes every 5s. Stops itself when the task reaches a terminal
status (done / failed / cancelled). Tails:
  - /api/training/progress  → epoch N/M + elapsed + ETA
  - /api/cluster/tasks      → cluster-side status + assigned worker
  - nvidia-smi (if available) → GPU utilization + VRAM, proxy for
                                "is the trainer actually computing?"
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_FILE = os.path.join(ROOT, ".env")


def _api_key() -> str:
    try:
        with open(API_KEY_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith("DASHBOARD_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


API_KEY = _api_key()


def _get(url: str, with_key: bool = False) -> dict | list | None:
    req = urllib.request.Request(url)
    if with_key and API_KEY:
        req.add_header("X-API-Key", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"__err__": str(exc)}


def _fmt_dur(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _gpu_status() -> str:
    """Return 'util%/temp/MB used' or '—' if nvidia-smi not available."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return "-"
        line = r.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return f"GPU {parts[0]}% / {parts[1]}C / {parts[2]}MB of {parts[3]}MB"
    except Exception:
        pass
    return "-"


def snapshot() -> bool:
    """Print one snapshot. Return True if task is still active."""
    ts = datetime.now().strftime("%H:%M:%S")
    # Progress endpoint
    prog = _get("http://127.0.0.1:5000/api/training/progress?include_terminal=1", with_key=True)
    tft_rec = None
    if isinstance(prog, dict) and prog.get("ok"):
        for t in prog.get("tasks") or []:
            if t.get("model") == "tft":
                tft_rec = t
                break
    # Cluster tasks
    tasks = _get("http://127.0.0.1:7700/api/cluster/tasks?limit=20")
    cluster_rec = None
    if isinstance(tasks, list):
        for t in tasks:
            if t.get("model_type") == "tft" and t.get("status") in (
                "running", "pending", "starting", "done", "failed", "cancelled"
            ):
                cluster_rec = t
                if t.get("status") == "running":
                    break  # prefer the running one
    # Workers
    workers = _get("http://127.0.0.1:7700/api/cluster/workers")
    busy_worker = None
    if isinstance(workers, list):
        for w in workers:
            if w.get("status") == "busy" and (w.get("current_task") or "").startswith(
                (cluster_rec or {}).get("task_id", "__none__")[:6]
            ):
                busy_worker = w
                break
    gpu = _gpu_status()

    print(f"\n[{ts}] " + "=" * 60)
    if cluster_rec:
        print(f"  cluster:  task={cluster_rec.get('task_id')}  "
              f"status={cluster_rec.get('status')}  "
              f"model={cluster_rec.get('model_type')}@{cluster_rec.get('timeframe')}  "
              f"assigned_to={(cluster_rec.get('assigned_to') or '-')[:8]}")
        if cluster_rec.get("started_at"):
            print(f"            started={cluster_rec.get('started_at','')[:19]}  "
                  f"last_update={cluster_rec.get('last_update_at','')[:19]}")
    else:
        print("  cluster:  no TFT task found")
    if tft_rec:
        cur = tft_rec.get("current_epoch") or 0
        tot = tft_rec.get("n_epochs") or 0
        el = tft_rec.get("elapsed_s")
        eta = tft_rec.get("eta_s")
        per_ep = tft_rec.get("mean_epoch_duration_s")
        bar = ""
        if tot:
            pct = min(100, int((cur / tot) * 100))
            blocks = int(pct / 5)
            # ASCII-only so Windows cp1252 stdout doesn't choke.
            bar = "#" * blocks + "-" * (20 - blocks) + f" {pct}%"
        print(f"  progress: epoch {cur}/{tot}  {bar}")
        print(f"            elapsed={_fmt_dur(el)}  "
              f"per-epoch={_fmt_dur(per_ep)}  "
              f"eta={_fmt_dur(eta)}  status={tft_rec.get('status')}")
    else:
        print("  progress: no entry yet (worker still loading data / building features)")
    if busy_worker:
        print(f"  worker:   {busy_worker.get('name')!r} on {busy_worker.get('hostname')!r}  "
              f"cpu={busy_worker.get('cpu_percent',0):.0f}%  "
              f"gpu={busy_worker.get('gpu_percent',0):.0f}%  "
              f"vram={busy_worker.get('gpu_mem_used_mb',0)}MB")
    print(f"  local gpu: {gpu}")

    # Stop when terminal
    active = True
    if tft_rec and tft_rec.get("status") in ("done", "error", "cancelled"):
        active = False
    elif cluster_rec and cluster_rec.get("status") in ("done", "failed", "cancelled"):
        active = False
    return active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="single snapshot then exit")
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()
    if args.once:
        snapshot()
        return
    print(f"Live monitor -- Ctrl+C to stop. Refreshing every {args.interval}s.")
    try:
        while True:
            active = snapshot()
            if not active:
                print("\n[monitor] task reached terminal state -- stopping.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[monitor] stopped by user.")


if __name__ == "__main__":
    main()
