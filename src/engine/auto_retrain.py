"""
auto_retrain — scheduled wrapper around the pipeline orchestrator.

Phase C of the institutional roadmap. Today retraining is manual:
operator clicks ▶ Run on the Pipeline Orchestrator card and waits.
Production institutional bots retrain continuously — every 24h or
sooner — so model drift can't quietly accumulate.

What this script does:
  1. Snapshot current WF Sharpe averages from latest_comparison.json +
     wf_results.json (the honest robustness signal — not in-sample).
  2. Run pipeline_orchestrator.run_pipeline() — full train + multi-TF
     backtest. Same code path the dashboard ▶ Run button uses.
  3. After completion, recompute WF Sharpe averages from the freshly
     written files and compare.
  4. If the new average is >= old (within --tolerance), the retrain is
     accepted: orchestrator status is 'done' and the new models are
     live. If new average is significantly worse, snapshot a regression
     report at data/retrain_regressions/<ts>.json and (optional --rollback)
     restore the model_meta files from the pre-run backup, leaving the
     prior models on disk.

Schedule via Windows Task Scheduler (existing local_scheduler.ps1):
  schtasks /Create /SC DAILY /ST 03:30 /TN AI-Trader-AutoRetrain
           /TR "powershell -File launch_auto_retrain.ps1"

Or one-shot from the dashboard's POST /api/auto_retrain/run.

Status writes to data/auto_retrain_status.json (filelock) so the
dashboard pill can show last-run time + verdict (accepted / regression).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
BACKUP_ROOT   = PROJECT_ROOT / "data" / "model_backups"
STATUS_PATH   = PROJECT_ROOT / "data" / "auto_retrain_status.json"
REGRESSION_DIR = PROJECT_ROOT / "data" / "retrain_regressions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(phase: str, message: str, **extra) -> None:
    sys.stderr.write(json.dumps({"phase": phase, "message": message,
                                  "ts": _now_iso(), **extra},
                                 default=str) + "\n")
    sys.stderr.flush()


def _wf_sharpe_snapshot() -> dict:
    """Read wf_results.json and aggregate WF Sharpe per strategy.
    Returns {strategy: mean_wf_sharpe}. Empty dict if no data."""
    wf_path = PROJECT_ROOT / "data" / "backtest" / "wf_results.json"
    if not wf_path.exists():
        return {}
    try:
        rows = json.loads(wf_path.read_text())
    except Exception:
        return {}
    sums: dict[str, list[float]] = {}
    for r in rows:
        s = (r.get("strategy") or "").strip()
        v = r.get("wf_mean_sharpe")
        if not s or v is None:
            continue
        try:
            sums.setdefault(s, []).append(float(v))
        except (TypeError, ValueError):
            continue
    return {s: round(sum(vals)/len(vals), 4) for s, vals in sums.items() if vals}


def _avg(d: dict) -> float | None:
    if not d:
        return None
    vals = [v for v in d.values() if v is not None]
    return round(sum(vals)/len(vals), 4) if vals else None


def _backup_models(label: str) -> Path | None:
    """Copy models/*.json (just metadata, not weights — those are big and
    we keep them in-place so partial-rollback semantics stay simple).
    Returns the backup directory path or None on failure."""
    if not MODELS_DIR.exists():
        return None
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = BACKUP_ROOT / f"{label}_{int(time.time())}"
    out_dir.mkdir(exist_ok=True)
    n = 0
    for p in MODELS_DIR.glob("*_meta.json"):
        try:
            shutil.copy2(p, out_dir / p.name)
            n += 1
        except Exception as exc:
            logger.warning("backup %s failed: %s", p.name, exc)
    logger.info("backed up %d meta files → %s", n, out_dir)
    return out_dir


def _restore_meta_from_backup(backup_dir: Path) -> int:
    """Restore _meta.json files from a backup. Returns count restored."""
    if not backup_dir or not backup_dir.exists():
        return 0
    n = 0
    for p in backup_dir.glob("*_meta.json"):
        try:
            shutil.copy2(p, MODELS_DIR / p.name)
            n += 1
        except Exception as exc:
            logger.warning("restore %s failed: %s", p.name, exc)
    return n


def _write_status(snap: dict) -> None:
    from src.utils.safe_json import write_json
    write_json(str(STATUS_PATH), snap)


def _record_regression(before: dict, after: dict, verdict: str) -> Path:
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = REGRESSION_DIR / f"{int(time.time())}.json"
    path.write_text(json.dumps({
        "ts": _now_iso(),
        "verdict": verdict,
        "before_avg":   _avg(before),
        "after_avg":    _avg(after),
        "before_per_strategy": before,
        "after_per_strategy":  after,
    }, indent=2, default=str), encoding="utf-8")
    return path


def run_auto_retrain(*,
                     tolerance: float = 0.05,
                     rollback: bool = False) -> dict:
    """End-to-end: snapshot, run, compare, optionally rollback. Returns a
    result dict shaped:
       {ok, verdict, before_avg, after_avg, delta, backup_dir, started_at, finished_at}
    verdict is one of:
       'accepted'    — new ≥ old × (1 - tolerance), models stay
       'regression'  — new < old × (1 - tolerance), regression recorded;
                       rollback applied iff rollback=True
       'no_baseline' — no prior wf_results to compare against (first run)
       'pipeline_error' — orchestrator returned status=error
    """
    started = time.time()
    started_iso = _now_iso()
    before = _wf_sharpe_snapshot()
    backup_dir = _backup_models("pre_retrain") if before else None
    _emit("auto_retrain", "starting",
          before_avg=_avg(before), tolerance=tolerance, rollback=rollback)

    # Run the pipeline. We import here to keep `auto_retrain` cheap to
    # import in tests / dashboard.
    from src.engine.pipeline_orchestrator import run_pipeline
    pipe_result = run_pipeline()
    if pipe_result.get("status") != "done":
        verdict = "pipeline_error"
        out = {
            "ok": False, "verdict": verdict,
            "before_avg": _avg(before), "after_avg": None, "delta": None,
            "backup_dir": str(backup_dir) if backup_dir else None,
            "started_at": started_iso, "finished_at": _now_iso(),
            "elapsed_s": round(time.time() - started, 1),
            "pipeline_status": pipe_result.get("status"),
            "pipeline_error":  (pipe_result.get("train") or {}).get("error")
                              or (pipe_result.get("backtest") or {}).get("error"),
        }
        _write_status(out)
        _emit("auto_retrain", "pipeline failed", **{k: out[k] for k in ("verdict","pipeline_error")})
        return out

    after = _wf_sharpe_snapshot()
    a_old = _avg(before)
    a_new = _avg(after)

    if a_old is None:
        verdict = "no_baseline"
        delta = None
    else:
        delta = round((a_new or 0) - (a_old or 0), 4)
        threshold = (a_old or 0) * (1 - tolerance) if (a_old or 0) > 0 else (a_old or 0) - tolerance
        verdict = "accepted" if (a_new or 0) >= threshold else "regression"

    out = {
        "ok": verdict in ("accepted", "no_baseline"),
        "verdict":     verdict,
        "before_avg":  a_old,
        "after_avg":   a_new,
        "delta":       delta,
        "tolerance":   tolerance,
        "backup_dir":  str(backup_dir) if backup_dir else None,
        "started_at":  started_iso,
        "finished_at": _now_iso(),
        "elapsed_s":   round(time.time() - started, 1),
    }

    if verdict == "regression":
        report = _record_regression(before, after, verdict)
        out["regression_report"] = str(report)
        if rollback and backup_dir:
            n = _restore_meta_from_backup(backup_dir)
            out["rollback_restored"] = n
            _emit("auto_retrain", "regression — rolled back",
                  delta=delta, restored=n, report=str(report))
        else:
            _emit("auto_retrain", "regression — kept new models (no rollback flag)",
                  delta=delta, report=str(report))
    else:
        _emit("auto_retrain", "accepted", before=a_old, after=a_new, delta=delta)

    _write_status(out)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Auto-retrain wrapper with regression guard")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Max acceptable WF Sharpe degradation (fraction). "
                         "0.05 means new must be >= old × 0.95.")
    ap.add_argument("--rollback", action="store_true",
                    help="If new WF Sharpe is below tolerance, restore "
                         "previous _meta.json files from backup.")
    args = ap.parse_args(argv)
    res = run_auto_retrain(tolerance=args.tolerance, rollback=args.rollback)
    print(json.dumps(res, default=str, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
