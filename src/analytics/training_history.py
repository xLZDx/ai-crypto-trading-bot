"""Phase D (2026-05-14) — Training run history + baseline tracking.

Every training run for every (model, tf) cell is recorded here with full
metric snapshot. The Analytics tab visualizes this. The CIO agent reads
it to make retrain decisions. Baselines let the operator compare each
new run against a stable reference point.

Storage: data/training_runs_history.json (atomic safe_json writes).

Why JSON not Parquet/DuckDB:
- Volume is small: ~5-50 runs/day, dict-per-row. JSON file stays <1 MB
  for years. Reading the whole file on each query is fine at that scale.
- safe_json + filelock gives us atomic-write + concurrent-read for free.
- A future migration to DuckDB is easy if/when row count justifies it.

Public surface
--------------
record_run(model, tf, **meta) -> run_id
    Append a row. If no baseline exists for (model, tf), this run is set
    as the baseline. Otherwise, delta_vs_baseline is computed against the
    current baseline.

get_runs(model=None, tf=None, limit=None) -> list[dict]
    Return runs filtered by model/tf, newest-first. limit caps the count.

get_baseline(model, tf) -> dict | None
    The current baseline row for the cell, or None if no runs yet.

promote_baseline(run_id) -> bool
    Mark a specific run as the new baseline for its (model, tf) cell.
    Deltas on subsequent runs are recomputed against the new baseline.

score_run(metrics) -> float
    Composite efficiency score, identical formula to the Phase B
    /api/training/rankings endpoint (anchored at 50 = random baseline).

winning_hyperparameters(model, tf) -> dict | None
    Across every run for (model, tf), return the hyperparameter set that
    produced the highest composite score.

backfill_from_meta_files() -> int
    One-shot bootstrap. Scans models/*_meta.json + models/_archived/
    and synthesizes history rows so the Analytics tab has something to
    render on first launch.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = _PROJECT_ROOT / "data" / "training_runs_history.json"
MODELS_DIR = _PROJECT_ROOT / "models"

# Mapping from canonical meta-JSON filenames to (model_key, default_tf).
_LEGACY_META_TO_CELL: dict[str, tuple[str, str]] = {
    "btc_rf_model_meta.json":         ("base", "1h"),
    "trend_model_meta.json":           ("trend", "1h"),
    "futures_short_model_meta.json":  ("futures", "1h"),
    "scalping_model_meta.json":       ("scalping", "1m"),
    "tft_model_meta.json":             ("tft", "1h"),
    "oft_model_meta.json":             ("oft", "1h"),
    "meta_labeler_meta.json":         ("meta", "1h"),
    "regime_classifier_meta.json":    ("regime", "1h"),
}

# Per-TF metas are named <model>_<tf>_meta.json. This regex matches and
# captures (model, tf).
_PER_TF_META_RE = re.compile(r"^([a-z]+)_(\d+[mhdw])_meta\.json$", re.IGNORECASE)


def _empty_state() -> dict:
    return {
        "schema_version": 1,
        "baselines": {},   # "<model>__<tf>" -> run_id
        "runs": [],        # list of run dicts, oldest -> newest
    }


def _load_state() -> dict:
    return read_json(str(HISTORY_PATH), default=_empty_state())


def _save_state(state: dict) -> None:
    write_json(str(HISTORY_PATH), state)


def _cell_key(model: str, tf: str) -> str:
    return f"{model}__{tf}"


def _to_pct(v: Any) -> float | None:
    """Normalize accuracy to percent (some trainers store 0.65, others 65)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f * 100.0 if 0.0 <= f <= 1.0 else f


def score_run(metrics: dict | None) -> float | None:
    """Composite efficiency score — identical formula to /api/training/rankings.

    0.5 * accuracy_test + 0.3 * (50 + (auc - 0.5)*100) + 0.2 * win_rate
    Renormalized over whichever of those three are present.
    """
    if not metrics:
        return None
    parts: list[tuple[float, float]] = []
    acc = metrics.get("accuracy_test") or metrics.get("accuracy")
    auc = metrics.get("auc_roc") or metrics.get("auc")
    wr = (metrics.get("bull_wr") or metrics.get("win_precision")
          or metrics.get("win_rate_pct"))
    if acc is not None:
        try:
            parts.append((float(acc), 0.5))
        except (TypeError, ValueError):
            pass
    if auc is not None:
        try:
            parts.append((50.0 + (float(auc) - 0.5) * 100.0, 0.3))
        except (TypeError, ValueError):
            pass
    if wr is not None:
        try:
            parts.append((float(wr), 0.2))
        except (TypeError, ValueError):
            pass
    if not parts:
        return None
    tw = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / tw


def _delta(curr_metrics: dict, baseline_metrics: dict) -> dict[str, float | None]:
    """Per-metric absolute delta (curr - baseline)."""
    def _safe(d: dict, k: str) -> float | None:
        v = d.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    keys = ("accuracy_test", "auc_roc", "win_precision", "bull_wr",
            "long_accuracy", "short_accuracy")
    out: dict[str, float | None] = {}
    for k in keys:
        c, b = _safe(curr_metrics, k), _safe(baseline_metrics, k)
        out[f"d_{k}"] = (c - b) if (c is not None and b is not None) else None
    s_c = score_run(curr_metrics)
    s_b = score_run(baseline_metrics)
    out["d_score"] = (s_c - s_b) if (s_c is not None and s_b is not None) else None
    return out


def _gen_run_id() -> str:
    return f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:6]}"


def record_run(
    model: str,
    tf: str,
    *,
    metrics: dict | None = None,
    hp: dict | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
    features_list: list[str] | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    trainer: str | None = None,
    model_path: str | None = None,
    meta_path: str | None = None,
    notes: str | None = None,
) -> str:
    """Append a run record. Returns the generated run_id.

    If no baseline exists for (model, tf), this run becomes the baseline
    (delta_vs_baseline = None). Otherwise delta is computed against the
    current baseline.
    """
    if not model or not tf:
        raise ValueError("model and tf are required")
    state = _load_state()
    run_id = _gen_run_id()
    metrics = metrics or {}
    # Normalize accuracy fields to percent.
    normalized = dict(metrics)
    for k in ("accuracy_test", "accuracy", "long_accuracy", "short_accuracy",
              "win_precision", "win_rate_pct", "bull_wr", "bear_wr",
              "walk_forward_mean_acc", "walk_forward_std_acc"):
        if k in normalized:
            normalized[k] = _to_pct(normalized[k])
    cell = _cell_key(model, tf)
    baseline_run_id = state["baselines"].get(cell)
    baseline_run = None
    if baseline_run_id:
        for r in reversed(state["runs"]):
            if r.get("run_id") == baseline_run_id:
                baseline_run = r
                break
    delta = _delta(normalized, baseline_run.get("metrics", {})) if baseline_run else None
    score = score_run(normalized)

    row = {
        "run_id": run_id,
        "model": model,
        "tf": tf,
        "cell": cell,
        "recorded_at": time.time(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": (finished_at - started_at)
                      if (started_at and finished_at and finished_at >= started_at)
                      else None,
        "n_samples": n_samples,
        "n_features": n_features,
        "features_list": list(features_list) if features_list else None,
        "hp": dict(hp) if hp else None,
        "metrics": normalized,
        "score": score,
        "is_baseline": baseline_run is None,
        "baseline_run_id": baseline_run_id,
        "delta_vs_baseline": delta,
        "trainer": trainer,
        "model_path": model_path,
        "meta_path": meta_path,
        "notes": notes,
    }
    state["runs"].append(row)
    if baseline_run is None:
        state["baselines"][cell] = run_id
    _save_state(state)
    logger.info("training_history: recorded %s for %s @ %s (score=%s)",
                run_id, model, tf, f"{score:.2f}" if score is not None else "n/a")
    return run_id


def get_runs(model: str | None = None, tf: str | None = None,
             limit: int | None = None) -> list[dict]:
    """Return runs filtered by model/tf, newest-first. limit caps the count."""
    state = _load_state()
    rows = list(state.get("runs") or [])
    if model:
        rows = [r for r in rows if r.get("model") == model]
    if tf:
        rows = [r for r in rows if r.get("tf") == tf]
    rows.reverse()  # newest first
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def get_baseline(model: str, tf: str) -> dict | None:
    """The baseline run row for (model, tf), or None if no runs recorded yet."""
    state = _load_state()
    cell = _cell_key(model, tf)
    rid = state["baselines"].get(cell)
    if not rid:
        return None
    for r in state.get("runs") or []:
        if r.get("run_id") == rid:
            return r
    return None


def promote_baseline(run_id: str) -> bool:
    """Mark `run_id` as the new baseline for its (model, tf) cell. Deltas
    on every later run for the same cell are recomputed."""
    state = _load_state()
    target = None
    for r in state.get("runs") or []:
        if r.get("run_id") == run_id:
            target = r
            break
    if not target:
        return False
    cell = target["cell"]
    state["baselines"][cell] = run_id
    # Mark each run in this cell with is_baseline + recompute deltas.
    baseline_metrics = target.get("metrics") or {}
    for r in state["runs"]:
        if r.get("cell") != cell:
            continue
        r["is_baseline"] = (r.get("run_id") == run_id)
        if r.get("run_id") == run_id:
            r["delta_vs_baseline"] = None
            r["baseline_run_id"] = None
        else:
            r["baseline_run_id"] = run_id
            r["delta_vs_baseline"] = _delta(r.get("metrics") or {}, baseline_metrics)
    _save_state(state)
    return True


def winning_hyperparameters(model: str, tf: str) -> dict | None:
    """Across every run for (model, tf), return the hp dict from the run
    with the highest composite score. None if there are zero runs."""
    rows = get_runs(model=model, tf=tf)
    if not rows:
        return None
    scored = [(r.get("score") or 0.0, r) for r in rows if r.get("hp") is not None]
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    best = scored[0][1]
    return {
        "best_run_id": best.get("run_id"),
        "best_score": best.get("score"),
        "best_hp": best.get("hp"),
        "n_runs_considered": len(scored),
    }


def backfill_from_meta_files() -> int:
    """One-shot. Scan models/*_meta.json + models/_archived/ and synthesize
    history rows. Idempotent — if a (model, tf, meta_path) tuple already
    has a recorded run with matching trainer+timestamp, skip it.
    """
    if not MODELS_DIR.exists():
        return 0
    state = _load_state()
    existing_paths = {(r.get("meta_path"), r.get("finished_at")) for r in state["runs"]}
    added = 0

    def _ingest(p: Path, model: str, tf: str) -> None:
        nonlocal added
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("backfill: skipping %s (read err: %s)", p, e)
            return
        # Pull a comparable timestamp out of the meta.
        ts_str = meta.get("last_trained") or meta.get("saved_at")
        finished_at: float | None = None
        if ts_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                finished_at = dt.timestamp()
            except Exception:
                pass
        # Dedup: if we already recorded this (meta_path, finished_at), skip.
        if (str(p), finished_at) in existing_paths:
            return
        metrics = {
            "accuracy_test": meta.get("accuracy"),
            "auc_roc": meta.get("auc_roc"),
            "long_accuracy": meta.get("long_accuracy"),
            "short_accuracy": meta.get("short_accuracy"),
            "win_precision": meta.get("win_precision"),
            "win_rate_pct": meta.get("win_rate_pct"),
            "walk_forward_mean_acc": meta.get("walk_forward_mean_acc"),
        }
        record_run(
            model=model, tf=tf,
            metrics=metrics,
            hp=meta.get("hp"),
            n_samples=meta.get("n_samples"),
            n_features=meta.get("n_features"),
            features_list=meta.get("features"),
            finished_at=finished_at,
            trainer=None,
            meta_path=str(p),
            notes="backfilled from meta",
        )
        added += 1
        # Refresh state for the next iteration (record_run reloads internally).

    # 1. Canonical legacy metas
    for fname, (model, tf) in _LEGACY_META_TO_CELL.items():
        p = MODELS_DIR / fname
        if p.exists():
            _ingest(p, model, tf)
    # 2. Per-TF metas
    for p in MODELS_DIR.glob("*_meta.json"):
        if p.name in _LEGACY_META_TO_CELL:
            continue
        m = _PER_TF_META_RE.match(p.name)
        if not m:
            continue
        _ingest(p, m.group(1).lower(), m.group(2).lower())

    return added


__all__ = [
    "HISTORY_PATH",
    "record_run", "get_runs", "get_baseline", "promote_baseline",
    "score_run", "winning_hyperparameters", "backfill_from_meta_files",
]
