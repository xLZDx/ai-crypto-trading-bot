"""PC load balancer — gives the dashboard a priority advantage over the
trading-pipeline workloads on this machine.

Why this exists
---------------
Operator 2026-05-15: "we have huge dashboard performance degradation when
training is running". On Windows the default process priority is NORMAL
for everything, so a 20-process training cluster + cluster_orch + worker
subprocesses CPU-starve the dashboard's Flask threads, making panels stall.

Goal
----
- The dashboard becomes the most-responsive workload (HIGH or
  ABOVE_NORMAL where the OS allows it).
- Training-pipeline siblings (cluster_orch, worker subprocesses) drop to
  BELOW_NORMAL so they yield CPU on contention.
- The trading bot is the **only** workload exempt from demotion when it
  is running against a live exchange (data/control.json trade_mode in
  LIVE_TRADE_MODES). Operator request: "the only exception is trading
  bot on live source".
- Idempotent + safe: never touch system / SYSTEM-owned processes, only
  touch processes whose `username()` matches the current user, and
  catch psutil.AccessDenied so a permission failure on one process never
  breaks the rest.

How it identifies what to manage
--------------------------------
1. `data/process_registry.json` — the canonical role table. Every
   long-running role (dashboard, bot, cluster_orch, orderbook_writer)
   claims here at startup.
2. cmdline pattern match — training subprocesses don't claim a role, but
   their cmdline starts with `python -m src.training.*` or the script
   names train_all_models.py / train_one_model.py / scalping_pipeline.py.

Live-trade exemption
--------------------
Reads `data/control.json` → `trade_mode`. Values in LIVE_TRADE_MODES mean
the bot must NOT be deprioritized — for safety, we don't risk slowing a
live trading loop. Testnet trade_mode is fair game (bot drops to NORMAL
not BELOW_NORMAL; we still don't demote it below dashboard).

Public surface
--------------
- `recommend_priorities(state=None) -> dict[pid, label]`
- `apply_priorities(*, dry_run=False) -> dict` returning what was changed
- `LB_STATE_PATH`, `is_enabled()`, `set_enabled(bool)`
- `start_background_thread(interval_s=30)` — single-fire idempotent

Operator-visible endpoint is in src/dashboard/app.py: /api/system/load_balancer.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LB_STATE_PATH = PROJECT_ROOT / "data" / "load_balancer_state.json"
CONTROL_PATH = PROJECT_ROOT / "data" / "control.json"
REGISTRY_PATH = PROJECT_ROOT / "data" / "process_registry.json"

# Trade modes that the bot exempts from demotion. control.json stores
# `trade_mode`; main.py sets it to one of these values when real money
# is in play.
LIVE_TRADE_MODES = frozenset({"real", "live", "mainnet", "prod", "production"})

# Friendly priority labels mapped to psutil constants. We use labels
# everywhere so test code can assert without importing psutil.
PRIORITY_LABELS = (
    "IDLE",          # below everything
    "BELOW_NORMAL",  # default for training siblings
    "NORMAL",        # default for everything else
    "ABOVE_NORMAL",  # default for dashboard
    "HIGH",          # operator can opt-in
)

# Default policy: which role gets which label. Keys are role names from
# process_registry.json; values are the label to apply.
DEFAULT_ROLE_POLICY: dict[str, str] = {
    "dashboard":         "ABOVE_NORMAL",
    "bot":               "NORMAL",          # may be overridden by trade-mode logic
    "cluster_orch":      "BELOW_NORMAL",
    "orderbook_writer":  "NORMAL",          # I/O-bound; demoting doesn't help
}

# Training subprocesses don't appear in the registry; classify them by
# cmdline. Order matters — first match wins.
TRAINING_CMDLINE_PATTERNS = (
    "train_all_models",
    "train_one_model",
    "scalping_pipeline",
    "src.training",
    "src.engine.cio_agent",
    "src.engine.auto_retrain",
    "binance_archive_downloader",
    "resample_ohlcv",
)
TRAINING_POLICY_LABEL = "BELOW_NORMAL"

# Lock for the background thread state file.
_lock = threading.RLock()
_bg_thread: threading.Thread | None = None
_bg_stop = threading.Event()


def _empty_state() -> dict:
    return {
        "schema_version": 1,
        "enabled": False,
        "last_run_ts": None,
        "last_run_summary": None,
        "policy": DEFAULT_ROLE_POLICY,
        "applied": {},   # pid -> {role, label, ts}
    }


def _load_state() -> dict:
    state = read_json(str(LB_STATE_PATH), default=None)
    if not isinstance(state, dict):
        return _empty_state()
    return state


def _save_state(state: dict) -> None:
    write_json(str(LB_STATE_PATH), state)


def is_enabled() -> bool:
    return bool(_load_state().get("enabled"))


def set_enabled(value: bool) -> bool:
    with _lock:
        s = _load_state()
        s["enabled"] = bool(value)
        _save_state(s)
    return bool(value)


def _label_to_psutil(label: str):
    """Translate a priority label to a psutil constant. Returns None if
    psutil isn't importable (load balancer becomes a no-op)."""
    try:
        import psutil
    except Exception:
        return None
    mapping = {
        "IDLE":         getattr(psutil, "IDLE_PRIORITY_CLASS", None),
        "BELOW_NORMAL": getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None),
        "NORMAL":       getattr(psutil, "NORMAL_PRIORITY_CLASS", None),
        "ABOVE_NORMAL": getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None),
        "HIGH":         getattr(psutil, "HIGH_PRIORITY_CLASS", None),
    }
    return mapping.get(label)


def _psutil_to_label(value) -> str | None:
    try:
        import psutil
    except Exception:
        return None
    rev = {
        getattr(psutil, "IDLE_PRIORITY_CLASS", None):         "IDLE",
        getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None): "BELOW_NORMAL",
        getattr(psutil, "NORMAL_PRIORITY_CLASS", None):       "NORMAL",
        getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None): "ABOVE_NORMAL",
        getattr(psutil, "HIGH_PRIORITY_CLASS", None):         "HIGH",
    }
    return rev.get(value)


def _get_trade_mode() -> str:
    ctl = read_json(str(CONTROL_PATH), default={})
    return str((ctl or {}).get("trade_mode") or "").lower()


def _is_live_trading() -> bool:
    return _get_trade_mode() in LIVE_TRADE_MODES


def _registry_roles() -> dict[int, str]:
    """Return {pid: role} for currently-registered roles."""
    data = read_json(str(REGISTRY_PATH), default={"roles": {}})
    out: dict[int, str] = {}
    for role, entry in ((data or {}).get("roles") or {}).items():
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        if isinstance(pid, int) and pid > 0:
            out[pid] = role
    return out


def _classify_process(proc, registered_roles: dict[int, str]) -> tuple[str | None, str]:
    """Return (role_or_None, classification_reason).

    role: 'dashboard'|'bot'|'cluster_orch'|'orderbook_writer'|'training' if known.
    """
    pid = proc.pid
    if pid in registered_roles:
        return (registered_roles[pid], "registry")
    try:
        cmdline = " ".join(proc.cmdline())
    except Exception:
        return (None, "cmdline_unavailable")
    cl_lower = cmdline.lower()
    for pat in TRAINING_CMDLINE_PATTERNS:
        if pat.lower() in cl_lower:
            return ("training", f"cmdline:{pat}")
    return (None, "unmatched")


def recommend_priorities(
    *,
    policy: dict[str, str] | None = None,
    live_trading: bool | None = None,
) -> list[dict]:
    """Walk current processes and return a planned-change list.
    Each entry: {pid, role, reason, current_label, recommended_label, exempt}

    Pure — does NOT change priorities. Use apply_priorities() for that.
    """
    try:
        import psutil
    except Exception:
        return []
    policy = dict(policy or DEFAULT_ROLE_POLICY)
    live = _is_live_trading() if live_trading is None else bool(live_trading)
    registered = _registry_roles()
    try:
        me_user = psutil.Process().username()
    except Exception:
        me_user = None

    plans: list[dict] = []
    for proc in psutil.process_iter(attrs=["pid", "name", "username"]):
        try:
            pid = proc.info.get("pid")
            if not pid or pid in (0, 4):
                continue
            # Only manage our own processes.
            uname = proc.info.get("username") or ""
            if me_user and uname and uname != me_user:
                continue
            role, reason = _classify_process(proc, registered)
            if role is None:
                continue
            # Pick recommended label.
            rec = policy.get(role) or (TRAINING_POLICY_LABEL if role == "training" else None)
            if rec is None:
                continue
            # Live-trade exemption: never demote the bot when live trading.
            exempt = False
            if role == "bot" and live:
                rec = "ABOVE_NORMAL"  # match dashboard so bot is never starved
                exempt = True
            current_label = None
            try:
                current_label = _psutil_to_label(proc.nice())
            except Exception:
                current_label = None
            plans.append({
                "pid": pid,
                "name": proc.info.get("name"),
                "role": role,
                "reason": reason,
                "current_label": current_label,
                "recommended_label": rec,
                "exempt_from_demotion": exempt,
            })
        except Exception:
            continue
    return plans


def apply_priorities(
    *,
    policy: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Compute recommendations + apply them. Returns a summary dict.

    summary = {
        "ts": float,
        "live_trading": bool,
        "dry_run": bool,
        "changed":     [pid, ...],
        "unchanged":   [pid, ...],
        "failed":      [{"pid":..., "error":...}],
        "skipped_protected_live_bot": bool,
        "plans":       [...],  # the full plan list (input data for clients)
    }
    """
    plans = recommend_priorities(policy=policy)
    summary: dict[str, Any] = {
        "ts": time.time(),
        "live_trading": _is_live_trading(),
        "dry_run": bool(dry_run),
        "changed": [],
        "unchanged": [],
        "failed": [],
        "skipped_protected_live_bot": False,
        "plans": plans,
    }
    try:
        import psutil
    except Exception:
        summary["error"] = "psutil_unavailable"
        return summary
    for plan in plans:
        pid = plan["pid"]
        rec_label = plan["recommended_label"]
        cur_label = plan.get("current_label")
        if cur_label == rec_label:
            summary["unchanged"].append(pid)
            continue
        if dry_run:
            summary["changed"].append(pid)
            continue
        rec_val = _label_to_psutil(rec_label)
        if rec_val is None:
            summary["failed"].append({"pid": pid, "error": f"unknown label {rec_label}"})
            continue
        try:
            psutil.Process(pid).nice(rec_val)
            summary["changed"].append(pid)
        except psutil.NoSuchProcess:
            summary["failed"].append({"pid": pid, "error": "no_such_process"})
        except psutil.AccessDenied as e:
            summary["failed"].append({"pid": pid, "error": f"access_denied: {e}"})
        except Exception as e:
            summary["failed"].append({"pid": pid, "error": f"{type(e).__name__}: {e}"})
    # Persist the run summary so the dashboard can show last-run state.
    with _lock:
        state = _load_state()
        state["last_run_ts"] = summary["ts"]
        state["last_run_summary"] = {
            k: v for k, v in summary.items() if k != "plans"
        }
        # Track applied label per pid.
        applied = state.get("applied") or {}
        for plan in plans:
            if plan["pid"] in summary["changed"] and not dry_run:
                applied[str(plan["pid"])] = {
                    "role":  plan["role"],
                    "label": plan["recommended_label"],
                    "ts":    summary["ts"],
                }
        state["applied"] = applied
        _save_state(state)
    return summary


def start_background_thread(interval_s: float = 30.0) -> bool:
    """Start (or no-op) a background applier loop. Returns True if a new
    thread was started, False if one was already running.

    The thread checks `is_enabled()` each tick — if disabled, it skips
    application but stays alive so the next /api/system/load_balancer/enable
    takes effect without restart.
    """
    global _bg_thread
    with _lock:
        if _bg_thread and _bg_thread.is_alive():
            return False
        _bg_stop.clear()

        def _loop():
            while not _bg_stop.is_set():
                try:
                    if is_enabled():
                        apply_priorities()
                except Exception as exc:
                    logger.warning("[load_balancer] tick failed: %s", exc)
                # Wait but wake on stop.
                _bg_stop.wait(interval_s)

        _bg_thread = threading.Thread(target=_loop, daemon=True,
                                       name="load-balancer")
        _bg_thread.start()
    return True


def stop_background_thread() -> None:
    _bg_stop.set()


__all__ = [
    "LB_STATE_PATH", "PRIORITY_LABELS",
    "DEFAULT_ROLE_POLICY", "LIVE_TRADE_MODES",
    "is_enabled", "set_enabled",
    "recommend_priorities", "apply_priorities",
    "start_background_thread", "stop_background_thread",
]
