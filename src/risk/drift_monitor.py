"""Phase 6b (2026-05-14) — hourly drift-poll background thread.

Why this exists
---------------
ml-engineer review note: "Not per inference tick (PSI on 30 features ×
every 1m tick = wasted CPU). Hourly poll on a background thread,
results cached in data/risk/drift_state.json; the ValidationGate reads
from cache."

This module owns the cache. It does NOT replace validators._check_drift
(which still computes drift on-demand when training a new model);
instead it provides the cached cross-cell view the dashboard surfaces
("which (model, tf) baselines have drifted recently?").

State file shape
----------------
data/risk/drift_state.json:
{
  "last_run_iso": "2026-05-14T01:00:00Z",
  "next_run_iso": "2026-05-14T02:00:00Z",
  "cells": [
    {"model": "trend", "tf": "1h", "baseline_age_days": 0.5,
     "report": {<DriftReport.to_dict()>}, "checked_at_iso": "..."},
    ...
  ]
}

Each iteration:
  1. Enumerate all data/risk/drift_baselines/*.json files.
  2. For each (model, tf), load the baseline. Source for the "live"
     feature distribution: the most recent training_runs/<model>__<tf>
     .parquet (or any parquet under data/parquet/ that has the trained
     feature columns). If no live snapshot is available the cell is
     marked "no_actual" and skipped.
  3. Run drift_psi.check_drift on the loaded actuals vs baseline.
  4. Persist the aggregated DriftReport per cell.

Operator can opt out via DRIFT_MONITOR_DISABLED=1.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = PROJECT_ROOT / "data" / "risk" / "drift_baselines"
STATE_FILE = PROJECT_ROOT / "data" / "risk" / "drift_state.json"
TRAINING_RUNS_DIR = PROJECT_ROOT / "data" / "training_runs"

# Poll interval — hourly per ml-engineer's recommended cadence.
DEFAULT_INTERVAL_S = 3600

# Hard upper bound: never run more than once every 10 min, even if a
# caller passes a smaller interval (avoids accidental DoS on the
# baselines directory).
MIN_INTERVAL_S = 600


@dataclass
class CellState:
    """One row of the drift_state.json's cells array."""
    model: str
    tf: str
    baseline_age_days: float | None = None
    checked_at_iso: str = ""
    actual_source: str = ""        # "training_runs" | "no_actual" | "skip"
    report: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    # P2: rolling count of consecutive hourly polls where pause_count > 0.
    # Enforce-tier features halt immediately (count not needed); confirm-tier
    # features halt only when this reaches 3.
    consecutive_pause_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tf": self.tf,
            "baseline_age_days": self.baseline_age_days,
            "checked_at_iso": self.checked_at_iso,
            "actual_source": self.actual_source,
            "report": self.report,
            "note": self.note,
            "consecutive_pause_count": self.consecutive_pause_count,
        }


def _parse_baseline_filename(stem: str) -> tuple[str, str] | None:
    """drift_baselines/<model>__<tf>.json → (model, tf), or None on parse fail."""
    if "__" not in stem:
        return None
    parts = stem.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _load_baseline(path: Path) -> dict[str, Any] | None:
    try:
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        feats = payload.get("features")
        if not isinstance(feats, dict):
            return None
        return {
            "features": feats,
            "saved_at": payload.get("saved_at"),
        }
    except Exception as e:
        logger.warning("[drift_monitor] could not load baseline %s: %s", path, e)
        return None


def _load_actual_sample(model: str, tf: str, baseline_features: dict[str, dict]) -> pd.DataFrame | None:
    """Try to load a recent feature sample for (model, tf).

    Data source priority:
      1. data/training_runs/<model>__<tf>.parquet — most recent training's
         live feature frame. Best signal of current training-window drift.
      2. None — no actual data available; the cell is marked "no_actual"
         and skipped in the run.

    Returns a DataFrame with at least the baseline's feature columns,
    or None when no usable source exists. Drops feature columns the
    actual snapshot doesn't have (graceful degradation)."""
    parquet_path = TRAINING_RUNS_DIR / f"{model}__{tf}.parquet"
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning("[drift_monitor] could not read %s: %s", parquet_path, e)
        return None
    if df is None or len(df) == 0:
        return None
    cols = [c for c in baseline_features.keys() if c in df.columns]
    if not cols:
        return None
    return df[cols].copy()


def _baseline_age_days(payload: dict) -> float | None:
    saved = payload.get("saved_at")
    if not saved:
        return None
    try:
        ts_clean = str(saved).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except Exception:
        return None


def _prev_consecutive_pause_count(model: str, tf: str) -> int:
    """Read the last persisted consecutive_pause_count for (model, tf) from
    the cached drift_state.json. Returns 0 if not found or on read error."""
    try:
        state = get_cached_state()
        for c in state.get("cells") or []:
            if c.get("model") == model and c.get("tf") == tf:
                return int(c.get("consecutive_pause_count") or 0)
    except Exception:
        pass
    return 0


def _run_one_cell(baseline_path: Path) -> CellState:
    """Run drift check for a single baseline file. Returns a CellState
    even on partial failure (so the state file still has the row)."""
    stem = baseline_path.stem
    parsed = _parse_baseline_filename(stem)
    if parsed is None:
        return CellState(model=stem, tf="?", note="filename-parse-failed",
                         actual_source="skip")
    model, tf = parsed

    payload = _load_baseline(baseline_path)
    if payload is None:
        return CellState(model=model, tf=tf, note="baseline-unreadable",
                         actual_source="skip",
                         checked_at_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    baseline_age = _baseline_age_days(payload)
    actual = _load_actual_sample(model, tf, payload["features"])
    checked = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if actual is None:
        return CellState(model=model, tf=tf,
                         baseline_age_days=baseline_age,
                         checked_at_iso=checked,
                         actual_source="no_actual",
                         note=f"no training_runs/{model}__{tf}.parquet")

    try:
        from src.risk.drift_psi import check_drift
        # Phase 6c — force warn mode here even if LLM_DRIFT_PAUSE=enforce.
        # The monitor is a passive observer and must persist the report
        # regardless of operator's enforcement mode; the bot's
        # is_drift_paused() consumer reads the persisted report and
        # honors enforce by halting trades. Pre-fix, drift_monitor inside
        # an enforce-mode env propagated DriftPauseError and the cell
        # got persisted as {error: "..."} instead of a real report,
        # which then made is_drift_paused unable to find pause_count.
        rep = check_drift(payload["features"], actual, force_mode="warn")

        # P2: track consecutive hourly polls with pause_count > 0 for confirm-tier.
        prev_count = _prev_consecutive_pause_count(model, tf)
        new_pause_count = rep.to_dict().get("pause_count", 0)
        consecutive = (prev_count + 1) if new_pause_count > 0 else 0

        return CellState(
            model=model, tf=tf,
            baseline_age_days=baseline_age,
            checked_at_iso=checked,
            actual_source="training_runs",
            report=rep.to_dict(),
            consecutive_pause_count=consecutive,
        )
    except Exception as e:
        # DriftPauseError or any other check_drift exception. Caller
        # already logged CRITICAL; we just persist the failure so the
        # dashboard can surface it.
        return CellState(
            model=model, tf=tf,
            baseline_age_days=baseline_age,
            checked_at_iso=checked,
            actual_source="training_runs",
            report={"error": f"{type(e).__name__}: {e}"},
            note="check_drift_raised",
        )


def run_once() -> dict[str, Any]:
    """Synchronous single pass over every baseline. Returns the new state
    dict and writes it to data/risk/drift_state.json. Exposed for the
    /api/drift/run endpoint (operator-driven refresh)."""
    cells: list[CellState] = []
    if BASELINES_DIR.exists():
        for path in sorted(BASELINES_DIR.glob("*.json")):
            cells.append(_run_one_cell(path))

    now = datetime.now(timezone.utc)
    state = {
        "last_run_iso": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cell_count": len(cells),
        "cells": [c.to_dict() for c in cells],
    }
    # Atomic write via safe_json — same pattern the rest of the project uses.
    try:
        from src.utils.safe_json import write_json
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(STATE_FILE), state, indent=2)
    except Exception as e:
        logger.warning("[drift_monitor] could not persist state: %s", e)
    return state


def get_cached_state() -> dict[str, Any]:
    """Read the cached drift_state.json. Used by the /api/drift/state
    endpoint to render the dashboard tile without recomputing."""
    try:
        from src.utils.safe_json import read_json
        return read_json(str(STATE_FILE), default={}) or {}
    except Exception as e:
        logger.warning("[drift_monitor] could not read cached state: %s", e)
        return {}


def is_drift_paused(model: str, tf: str) -> tuple[bool, str]:
    """Return (is_paused, reason) for the given (model, tf) cell.

    P2 two-tier enforcement (2026-05-17):
      - Enforce-tier (DRIFT_ENFORCE_FEATURES): halt on first poll with
        PSI >= 0.25 on any matching feature. No consecutive requirement.
      - Confirm-tier (remaining DRIFT_HARD_FEATURES): halt only after
        consecutive_pause_count >= 3 (three consecutive hourly polls).
        Single-spike false positives are ignored.

    Both tiers require LLM_DRIFT_PAUSE=enforce. Default 'warn' = advisory.

    Usage in the bot:
        from src.risk.drift_monitor import is_drift_paused
        paused, why = is_drift_paused('trend', '1h')
        if paused:
            logger.warning("[trade] skipped trend@1h signal — drift: %s", why)
            return  # don't trade
    """
    mode = (os.environ.get("LLM_DRIFT_PAUSE") or "warn").strip().lower()
    if mode != "enforce":
        return False, f"LLM_DRIFT_PAUSE={mode} — not enforcing"

    state = get_cached_state()
    cells = state.get("cells") or []
    if not cells:
        return False, "no_baselines_yet"

    for c in cells:
        if c.get("model") != model or c.get("tf") != tf:
            continue

        rep = c.get("report") or {}
        if not rep or rep.get("pause_count", 0) == 0:
            return False, "cell_clean"

        findings = rep.get("findings") or []

        # Enforce-tier: immediate halt on first poll.
        from src.risk.drift_psi import _enforce_features
        enforce_set = _enforce_features()
        enforce_paused = [
            f.get("feature") for f in findings
            if f.get("severity") == "pause"
            and f.get("is_enforce")
            and f.get("feature") in enforce_set
        ]
        if enforce_paused:
            names = ", ".join(enforce_paused[:5])
            return True, f"enforce-tier drift (immediate): {names}"

        # Confirm-tier: halt only after 3 consecutive polls.
        consecutive = int(c.get("consecutive_pause_count") or 0)
        if consecutive >= 3:
            confirm_paused = [
                f.get("feature") for f in findings
                if f.get("severity") == "pause" and f.get("is_hard")
                and not f.get("is_enforce")
            ]
            names = ", ".join(confirm_paused[:5]) or "?"
            return True, (
                f"confirm-tier drift ({consecutive} consecutive polls): {names}"
            )
        return False, (
            f"confirm-tier drift detected but only {consecutive}/3 consecutive polls"
        )

    return False, "cell_not_found"


# ── Background thread ────────────────────────────────────────────────────
_thread: threading.Thread | None = None
_stop = threading.Event()
_thread_lock = threading.Lock()


def _loop(interval_s: int) -> None:
    """Background loop body. Runs once immediately, then sleeps interval_s."""
    while not _stop.is_set():
        try:
            run_once()
        except Exception as e:
            logger.warning("[drift_monitor] run_once raised: %s", e)
        # Sleep in 5-second slices so stop() is responsive.
        slept = 0
        while slept < interval_s and not _stop.is_set():
            time.sleep(min(5, interval_s - slept))
            slept += 5


def start(interval_s: int = DEFAULT_INTERVAL_S) -> bool:
    """Start the hourly drift-poll thread. Idempotent. Returns True iff
    the thread was started by this call; False if it was already running
    or DRIFT_MONITOR_DISABLED is set."""
    if (os.environ.get("DRIFT_MONITOR_DISABLED") or "").strip() in ("1", "true", "yes"):
        logger.info("[drift_monitor] disabled via DRIFT_MONITOR_DISABLED")
        return False
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return False
        interval_s = max(MIN_INTERVAL_S, int(interval_s))
        _stop.clear()
        _thread = threading.Thread(
            target=_loop, args=(interval_s,),
            name="DriftMonitor-Hourly", daemon=True,
        )
        _thread.start()
        logger.info("[drift_monitor] thread started (interval=%ds)", interval_s)
        return True


def stop() -> None:
    """Signal the background thread to exit. Idempotent."""
    _stop.set()


__all__ = [
    "CellState", "DEFAULT_INTERVAL_S",
    "run_once", "get_cached_state", "is_drift_paused", "start", "stop",
]
