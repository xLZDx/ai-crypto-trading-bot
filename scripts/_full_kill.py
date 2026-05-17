"""Hard reset — kill EVERY python process related to this app.

2026-05-15 operator request: "kill all python, training bot or any related
processes to this app". Matches by:
  - cmdline contains the project path 'AI trading assistance', OR
  - cmdline contains any of our top-level module paths (src.dashboard,
    src.main, src.training, src.engine, src.data_governance,
    src.data_ingestion, src.utils.load_balancer), OR
  - cmdline contains 'joblib' / 'loky' (their parent is one of ours)

Excludes:
  - The current Python process and its parent (so we don't kill ourselves).
"""
from __future__ import annotations

import os
import sys
import time

import psutil

PROJECT_PATH_TOKENS = (
    "ai trading assistance",
    "src.dashboard",
    "src.main",
    "src.training",
    "src.engine",
    "src.data_governance",
    "src.data_ingestion",
    "src.utils.load_balancer",
    "src.server.control_plane",
    "src.analytics",
    "scalping_pipeline",
    "train_all_models",
    "train_one_model",
    "binance_archive_downloader",
)

# Joblib / loky child processes are also fair game — they share the parent's
# fate. They're keyed by parent cmdline rather than their own.
LOKY_TOKENS = ("joblib.externals.loky", "loky.backend")


def matches(cmd: str) -> str | None:
    """Return reason if cmd should be killed, else None."""
    cl = cmd.lower()
    for tok in PROJECT_PATH_TOKENS:
        if tok in cl:
            return f"project_token:{tok}"
    for tok in LOKY_TOKENS:
        if tok in cl:
            return f"loky_token:{tok}"
    return None


def main() -> None:
    my_pid = os.getpid()
    my_parent = psutil.Process(my_pid).ppid()
    my_user = psutil.Process().username()
    self_protected = {my_pid, my_parent}

    plan: list[tuple[int, str, str, float]] = []
    for p in psutil.process_iter(["pid", "name", "username", "cmdline", "create_time"]):
        try:
            if p.info["username"] != my_user:
                continue
            if p.info["name"] != "python.exe":
                continue
            if p.info["pid"] in self_protected:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            reason = matches(cmd)
            if reason:
                plan.append((p.info["pid"], reason, cmd[:120],
                             (time.time() - p.info["create_time"]) / 60))
        except Exception:
            continue

    print(f"=== KILL PLAN -- {len(plan)} pids ===")
    for pid, reason, cmd, age in plan:
        print(f"  {pid:>6}  age={age:6.1f}min  {reason:30s}  {cmd}")

    if "--execute" not in sys.argv:
        print("\n[dry-run] pass --execute to kill")
        return

    print()
    print("=== EXECUTING ===")
    killed = already = failed = 0
    # Kill children first (loky workers) so their parents don't respawn them.
    # We can approximate that by reversing the order — newer pids are usually
    # children in our case. Better: use process trees.
    for pid, reason, cmd, _ in plan:
        try:
            proc = psutil.Process(pid)
            proc.kill()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                pass
            print(f"  killed {pid} ({reason})")
            killed += 1
        except psutil.NoSuchProcess:
            already += 1
        except psutil.AccessDenied as e:
            print(f"  FAILED {pid}: access_denied: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAILED {pid}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"=== SUMMARY === killed={killed} already_dead={already} failed={failed}")


if __name__ == "__main__":
    main()
