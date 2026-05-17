"""One-shot zombie killer — 2026-05-15 operator request.

Aggressive scope: kill orphan loky workers, duplicate orchestrators, AND
collapse per-(model, tf) training duplicates to ONE process (the oldest).

Protected: dashboard, bot, orderbook_writer.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import psutil

PROTECTED_PIDS = {49968, 8020, 22112}  # dashboard, bot, orderbook_writer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "data" / "_kill_plan.json"

RE_PARENT_PID = re.compile(r"parent_pid=(\d+)")
RE_TF = re.compile(r"timeframe='([^']+)'")
RE_TRAIN = re.compile(r"src\.engine\.(train_\w+)")


def classify(cmd: str) -> str:
    if "src.training.distributed.orchestrator" in cmd:
        return "cluster_orch"
    if "src.data_governance.orchestrator" in cmd:
        return "data_gov_orch"
    if "src.dashboard.app" in cmd:
        return "dashboard"
    norm = cmd.replace("\\", "/")
    if "src/main.py" in norm:
        return "bot"
    if "orderbook_parquet_writer" in cmd:
        return "orderbook_writer"
    if "loky" in cmd:
        return "loky_worker"
    m = RE_TRAIN.search(cmd)
    if m:
        tf_m = RE_TF.search(cmd)
        tf = tf_m.group(1) if tf_m else "?"
        return f"train::{m.group(1)}::{tf}"
    return "other"


def compute_plan() -> list[tuple[int, str, str, float]]:
    """Return [(pid, reason, detail, age_min)]."""
    my_user = psutil.Process().username()
    procs: dict[int, dict] = {}
    for p in psutil.process_iter(["pid", "name", "username", "cmdline",
                                   "create_time", "ppid"]):
        try:
            if p.info["username"] != my_user:
                continue
            if p.info["name"] != "python.exe":
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            procs[p.pid] = {
                "pid": p.pid,
                "ppid": p.info["ppid"],
                "cmd": cmd,
                "create_time": p.info["create_time"],
                "kind": classify(cmd),
            }
        except Exception:
            continue

    kill: list[tuple[int, str, str, float]] = []
    now = time.time()

    # 1) Orphan loky workers — declared parent_pid in cmdline is dead.
    for rec in procs.values():
        if rec["kind"] != "loky_worker":
            continue
        m = RE_PARENT_PID.search(rec["cmd"])
        declared = int(m.group(1)) if m else None
        parent_alive = bool(declared and psutil.pid_exists(declared))
        if not parent_alive:
            kill.append((rec["pid"], "orphan_loky",
                         f"declared_parent={declared} dead",
                         (now - rec["create_time"]) / 60))

    # 2) Duplicate cluster_orch: only the actual port-7700 holder survives.
    port_holder: int | None = None
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == 7700 and c.status == psutil.CONN_LISTEN:
                port_holder = c.pid
                break
    except (psutil.AccessDenied, Exception):
        pass
    co = [r for r in procs.values() if r["kind"] == "cluster_orch"]
    if port_holder is None and len(co) > 1:
        # Fall back: keep oldest.
        co.sort(key=lambda r: r["create_time"])
        port_holder = co[0]["pid"]
    for r in co:
        if r["pid"] != port_holder:
            kill.append((r["pid"], "duplicate_cluster_orch",
                         f"not port-7700 holder (holder={port_holder})",
                         (now - r["create_time"]) / 60))

    # 3) Duplicate data_governance.orchestrator: keep oldest.
    dgo = sorted([r for r in procs.values() if r["kind"] == "data_gov_orch"],
                 key=lambda r: r["create_time"])
    if len(dgo) > 1:
        keep_pid = dgo[0]["pid"]
        for r in dgo[1:]:
            kill.append((r["pid"], "duplicate_data_gov_orch",
                         f"keeping oldest pid={keep_pid}",
                         (now - r["create_time"]) / 60))

    # 4) AGGRESSIVE: per-(model, tf) training duplicates — keep oldest.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in procs.values():
        if r["kind"].startswith("train::"):
            buckets[r["kind"]].append(r)
    for key, group in buckets.items():
        if len(group) <= 1:
            continue
        group.sort(key=lambda r: r["create_time"])
        keep = group[0]
        for r in group[1:]:
            kill.append((r["pid"], f"duplicate_{key}",
                         f"keeping oldest pid={keep['pid']} "
                         f"({(now - keep['create_time'])/60:.1f}min old)",
                         (now - r["create_time"]) / 60))

    # Protection.
    kill = [k for k in kill if k[0] not in PROTECTED_PIDS]
    return kill, procs


def main() -> None:
    plan, procs = compute_plan()
    print(f"=== PLAN -- {len(plan)} kills planned ===")
    print(f"{'PID':>6}  {'AGE_MIN':>8}  REASON")
    for pid, reason, detail, age in plan:
        print(f"{pid:>6}  {age:8.1f}  {reason:30s}  {detail}")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(
        {"kill": [{"pid": p, "reason": r, "detail": d, "age_min": a}
                  for p, r, d, a in plan],
         "protected": sorted(PROTECTED_PIDS)}, indent=2), encoding="utf-8")
    print(f"plan written to {PLAN_PATH}")

    if "--execute" not in sys.argv:
        print("[dry-run] pass --execute to actually kill")
        return

    print()
    print("=== EXECUTING ===")
    results = []
    for pid, reason, detail, age in plan:
        if pid in PROTECTED_PIDS:
            results.append((pid, "skipped_protected"))
            continue
        try:
            p = psutil.Process(pid)
            p.kill()
            try:
                p.wait(timeout=3)
            except psutil.TimeoutExpired:
                pass
            results.append((pid, "killed"))
            print(f"  killed pid={pid} ({reason})")
        except psutil.NoSuchProcess:
            results.append((pid, "already_dead"))
        except psutil.AccessDenied as e:
            results.append((pid, f"access_denied: {e}"))
            print(f"  FAILED pid={pid}: access denied")
        except Exception as e:
            results.append((pid, f"err: {type(e).__name__}: {e}"))
            print(f"  FAILED pid={pid}: {e}")

    summary = {
        "killed":            sum(1 for _, r in results if r == "killed"),
        "already_dead":      sum(1 for _, r in results if r == "already_dead"),
        "skipped_protected": sum(1 for _, r in results if r == "skipped_protected"),
        "failed":            sum(1 for _, r in results
                                 if r not in {"killed", "already_dead",
                                              "skipped_protected"}),
    }
    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
