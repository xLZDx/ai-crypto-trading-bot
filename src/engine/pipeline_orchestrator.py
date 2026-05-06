"""
pipeline_orchestrator — sequential train → multi-TF backtest runner.

Designed to be spawned as a long-lived subprocess (Popen) so the dashboard
process can't be brought down by training memory pressure. The orchestrator
writes a status file at `data/pipeline_status.json` (filelock + atomic
write) so the dashboard pill stays observable across restarts.

Phases:
  1. train  — runs `train_all()` (per-key multi-TF iteration)
  2. backtest — runs `run_full_backtest(timeframes=BACKTEST_TFS)` so the
                Stability heatmap (PR 4) gets real WF Sharpe data across
                all TFs, not just 1h.

Usage (CLI):
    python -m src.engine.pipeline_orchestrator
    python -m src.engine.pipeline_orchestrator --skip-train
    python -m src.engine.pipeline_orchestrator --backtest-tfs 5m,1h,4h,1d

Usage (programmatic):
    from src.engine.pipeline_orchestrator import run_pipeline
    run_pipeline()

Status file shape (data/pipeline_status.json):
    {
      "status":         "running"|"done"|"error"|"idle",
      "phase":          "train"|"backtest"|null,
      "started_at":     ISO 8601 UTC,
      "finished_at":    ISO 8601 UTC | null,
      "elapsed_s":      float,
      "train": {
          "started_at": ..., "finished_at": ..., "ok": bool, "error": str|null
      },
      "backtest": {
          "started_at": ..., "finished_at": ..., "ok": bool,
          "rows_written": int, "timeframes": [...], "error": str|null
      },
      "last_event":     {"phase": str, "message": str, "ts": ISO}
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH  = PROJECT_ROOT / "data" / "pipeline_status.json"

# Default multi-TF set for the post-training backtest. Excludes 1m (too
# noisy for stability comparison) and 1mo (too few bars for WF). 1w is
# borderline but kept because the heatmap loses its swing-TF column without
# it.
DEFAULT_BACKTEST_TFS: tuple[str, ...] = ("5m", "1h", "4h", "1d", "1w")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_event(phase: str, message: str, **extra) -> None:
    ev = {"phase": phase, "message": message, "ts": _now_iso(), **extra}
    sys.stderr.write(json.dumps(ev, default=str) + "\n")
    sys.stderr.flush()


def _read_status() -> dict:
    from src.utils.safe_json import read_json
    return read_json(str(STATUS_PATH), default={}) or {}


def _write_status(snap: dict) -> None:
    from src.utils.safe_json import write_json
    write_json(str(STATUS_PATH), snap)


def _update_status(**patch) -> dict:
    snap = _read_status()
    snap.update(patch)
    _write_status(snap)
    return snap


def _run_train_phase() -> dict:
    """Phase 1: full multi-TF training. Returns {ok, error}.

    Catches any uncaught exception so the orchestrator can record it and
    proceed to backtest anyway — partial training is still useful (e.g.
    if TFT GPU OOM but tabular models trained fine). Backtest doesn't
    depend on every model existing; missing models just fall through to
    pure signals.
    """
    started = time.time()
    _emit_event("train", "starting train_all (multi-TF)")
    _update_status(phase="train",
                   train={"started_at": _now_iso(), "ok": False,
                          "finished_at": None, "error": None})
    try:
        from src.engine.train_all_models import train_all
        train_all()
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": True, "error": None,
               "elapsed_s": round(time.time() - started, 1)}
        _emit_event("train", "train_all completed",
                    elapsed_s=out["elapsed_s"])
        return out
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("train phase failed: %s\n%s", exc, tb)
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": False,
               "error": f"{type(exc).__name__}: {exc}",
               "elapsed_s": round(time.time() - started, 1)}
        _emit_event("train", "train_all FAILED",
                    error=out["error"], elapsed_s=out["elapsed_s"])
        return out


def _run_backtest_phase(timeframes: tuple[str, ...]) -> dict:
    """Phase 2: multi-TF backtest to populate Stability heatmap. Returns
    {ok, error, rows_written, timeframes}."""
    started = time.time()
    _emit_event("backtest", "starting run_full_backtest",
                timeframes=list(timeframes))
    _update_status(phase="backtest",
                   backtest={"started_at": _now_iso(), "ok": False,
                             "finished_at": None, "error": None,
                             "rows_written": 0,
                             "timeframes": list(timeframes)})
    try:
        from src.engine.backtester import run_full_backtest
        df = run_full_backtest(timeframes=timeframes)
        rows = int(len(df)) if df is not None else 0
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": True, "error": None, "rows_written": rows,
               "timeframes": list(timeframes),
               "elapsed_s": round(time.time() - started, 1)}
        _emit_event("backtest", f"run_full_backtest completed ({rows} rows)",
                    rows=rows, elapsed_s=out["elapsed_s"])
        return out
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("backtest phase failed: %s\n%s", exc, tb)
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": False,
               "error": f"{type(exc).__name__}: {exc}",
               "rows_written": 0, "timeframes": list(timeframes),
               "elapsed_s": round(time.time() - started, 1)}
        _emit_event("backtest", "run_full_backtest FAILED",
                    error=out["error"], elapsed_s=out["elapsed_s"])
        return out


def run_pipeline(*,
                 skip_train: bool = False,
                 skip_backtest: bool = False,
                 backtest_tfs: tuple[str, ...] = DEFAULT_BACKTEST_TFS) -> dict:
    """Run the orchestration end-to-end. Idempotent — safe to re-run.

    Always writes a complete status file at the end so the dashboard pill
    transitions cleanly from running → done (or error) regardless of which
    sub-phase failed.
    """
    started = time.time()
    started_iso = _now_iso()
    _write_status({
        "status":      "running",
        "phase":       None,
        "started_at":  started_iso,
        "finished_at": None,
        "elapsed_s":   0.0,
        "train":       None,
        "backtest":    None,
        "last_event":  None,
    })
    _emit_event("pipeline", "started",
                skip_train=skip_train, skip_backtest=skip_backtest,
                backtest_tfs=list(backtest_tfs))

    train_result = ({"skipped": True, "ok": True, "error": None}
                    if skip_train else _run_train_phase())
    backtest_result = ({"skipped": True, "ok": True, "error": None,
                        "rows_written": 0,
                        "timeframes": list(backtest_tfs)}
                       if skip_backtest else _run_backtest_phase(backtest_tfs))

    elapsed = round(time.time() - started, 1)
    overall_ok = train_result.get("ok", False) and backtest_result.get("ok", False)
    snap = {
        "status":      "done" if overall_ok else "error",
        "phase":       None,
        "started_at":  started_iso,
        "finished_at": _now_iso(),
        "elapsed_s":   elapsed,
        "train":       train_result,
        "backtest":    backtest_result,
        "last_event":  {"phase": "pipeline",
                        "message": ("done" if overall_ok else "error"),
                        "ts": _now_iso()},
    }
    _write_status(snap)
    _emit_event("pipeline", "finished",
                ok=overall_ok, elapsed_s=elapsed)
    return snap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description="Pipeline orchestrator: train → multi-TF backtest")
    ap.add_argument("--skip-train",    action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--backtest-tfs", type=str,
                    default=",".join(DEFAULT_BACKTEST_TFS),
                    help="Comma-separated list of timeframes for the multi-TF backtest.")
    args = ap.parse_args(argv)
    tfs = tuple(s.strip() for s in args.backtest_tfs.split(",") if s.strip())
    snap = run_pipeline(skip_train=args.skip_train,
                        skip_backtest=args.skip_backtest,
                        backtest_tfs=tfs)
    print(json.dumps(snap, default=str, indent=2))
    return 0 if snap.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
