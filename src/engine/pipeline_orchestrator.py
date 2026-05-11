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


# ─── Phase 100e — cluster dispatch helpers ────────────────────────────────
#
# Pipeline orchestrator used to call train_all() in-process, which ran every
# trainer locally and left the distributed cluster (LOCAL_RAZER_CPU/GPU +
# IVAN_CPU/GPU) idle. Operator screenshot 2026-05-11 02:55: all 4 online
# workers status=idle while pipeline_orchestrator was burning Razer CPU
# solo. Workers were never asked to do anything.
#
# This module now submits each (model, tf) cell to the cluster orchestrator
# (port 7700) and chains a backtest_cell task after each train succeeds.
# Multiple cells run in parallel across all 4 lanes; per-cell sequencing
# (train must complete before BT submits) is strict.
#
# Emergency rollback: AI_TRADER_PIPELINE_LOCAL=1 reverts to in-process
# train_all() — kept as a safety hatch while the cluster path beds in.

import os as _os
import urllib.request as _urlreq
import urllib.error as _urlerr

CLUSTER_BASE_URL = _os.getenv('AI_TRADER_CLUSTER_URL', 'http://127.0.0.1:7700')

# Dashboard model_key → cluster worker model_type mapping. Must stay in sync
# with src/dashboard/app.py:_DASH_TO_CLUSTER_KEY. Diverging-name keys only;
# rest fall through.
_PIPELINE_DASH_TO_CLUSTER: dict[str, str] = {
    'base':     'btc_rf',
    'futures':  'futures_short',
    'meta':     'meta_labeler',
}


def _to_cluster_model_type(dash_key: str) -> str:
    return _PIPELINE_DASH_TO_CLUSTER.get(dash_key, dash_key)


def _cluster_post(path: str, body: dict, timeout: float = 15.0) -> tuple[dict, int]:
    """POST to the cluster orchestrator. Returns (json_body, http_status).
    503 on connection failures so the caller can treat cluster-unreachable
    the same as 503."""
    try:
        data = json.dumps(body or {}).encode('utf-8')
        req = _urlreq.Request(f'{CLUSTER_BASE_URL}{path}', data=data,
                              method='POST',
                              headers={'Content-Type': 'application/json'})
        with _urlreq.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except (_urlerr.URLError, OSError, json.JSONDecodeError) as exc:
        return {'error': f'cluster unreachable: {exc}'}, 503


def _cluster_get(path: str, timeout: float = 15.0) -> tuple[object, int]:
    try:
        with _urlreq.urlopen(f'{CLUSTER_BASE_URL}{path}', timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8')), r.status
    except (_urlerr.URLError, OSError, json.JSONDecodeError) as exc:
        return {'error': f'cluster unreachable: {exc}'}, 503


def _pipeline_cell_list() -> list[tuple[str, str]]:
    """Build (model, tf) cell list in model-major order from
    DEFAULT_PER_KEY_TFS. Same order/source as the dashboard's
    _retrain_all_cell_list — single source of truth for "what cells exist".
    """
    from src.engine.train_all_models import DEFAULT_PER_KEY_TFS
    cells: list[tuple[str, str]] = []
    for model_key in DEFAULT_PER_KEY_TFS:
        for tf in DEFAULT_PER_KEY_TFS[model_key]:
            cells.append((model_key, tf))
    return cells


def _pipeline_build_train_spec(model_key: str, tf: str) -> dict:
    return {
        'model_type':  _to_cluster_model_type(model_key),
        'symbol':      'BTC/USDT',
        'timeframe':   tf,
        'config':      {},
        'data_path':   '',
        'output_path': '',
    }


def _pipeline_build_bt_spec(model_key: str, tf: str) -> dict:
    return {
        'model_type': 'backtest_cell',
        'symbol':     'BTC_USDT',   # underscore form per worker.py contract
        'timeframe':  tf,
        'config': {
            'initial_capital': 10000.0,
            'fee_preset':      'futures',
            'models':          [_to_cluster_model_type(model_key)],
        },
    }


def _pipeline_step(cells: list[tuple[str, str]], state: dict,
                    by_id: dict, submit_fn) -> dict:
    """Pure per-iteration step. Same shape as
    src/dashboard/app.py:_retrain_all_step — when a train cell turns 'done',
    submit_fn is called to dispatch its BT; failed/cancelled trains skip BT.
    Mutates state['train_tids']/['bt_tids']/['train_done']/['bt_done']/
    ['cell_errors'] in place. Returns aggregate counters + finished flag.
    """
    train_tids   = state['train_tids']
    bt_tids      = state['bt_tids']
    train_done   = state['train_done']
    bt_done      = state['bt_done']
    cell_errors  = state['cell_errors']
    for cell, tid in train_tids.items():
        if cell in train_done:
            continue
        task = by_id.get(tid)
        if not task:
            continue
        s = task.get('status')
        if s == 'done':
            train_done.add(cell)
            m, tf = cell
            bt_task_id = submit_fn(m, tf, _pipeline_build_bt_spec(m, tf))
            if bt_task_id:
                bt_tids[cell] = bt_task_id
            else:
                cell_errors[cell] = 'BT submit failed (cluster unreachable)'
                bt_done.add(cell)
        elif s in ('failed', 'cancelled'):
            train_done.add(cell)
            bt_done.add(cell)
            cell_errors[cell] = f'train {s}: {task.get("error", "")[:120]}'
    for cell, tid in bt_tids.items():
        if cell in bt_done:
            continue
        task = by_id.get(tid)
        if not task:
            continue
        s = task.get('status')
        if s in ('done', 'failed', 'cancelled'):
            bt_done.add(cell)
            if s != 'done':
                cell_errors[cell] = f'BT {s}: {task.get("error", "")[:120]}'
    cells_complete = sum(1 for c in cells if c in train_done and c in bt_done)
    train_inflight = sum(1 for c in train_tids if c not in train_done)
    bt_inflight    = sum(1 for c in bt_tids   if c not in bt_done)
    finished       = all(c in train_done and c in bt_done for c in cells)
    return {
        'cells_complete': cells_complete,
        'train_inflight': train_inflight,
        'bt_inflight':    bt_inflight,
        'finished':       finished,
    }


def _run_train_phase_cluster() -> dict:
    """Phase 100e — distributed train phase. Submits every (model, tf)
    cell from DEFAULT_PER_KEY_TFS to the cluster orchestrator; the
    orchestrator routes to LOCAL_RAZER_CPU/GPU + IVAN_CPU/GPU lanes.
    Per cell: train must complete before its BT submits. Across cells:
    full parallelism across all healthy worker lanes — Ivan and Razer
    BOTH get tasks instead of Ivan staying idle while Razer burns CPU.
    """
    started = time.time()
    cells = _pipeline_cell_list()
    _emit_event("train", "cluster dispatch starting",
                cells_total=len(cells))
    _update_status(phase="train",
                   train={"started_at": _now_iso(), "ok": False,
                          "finished_at": None, "error": None,
                          "cells_total": len(cells)})

    train_tids: dict[tuple[str, str], str] = {}
    submit_failures: list[str] = []
    for (m, tf) in cells:
        body, http_status = _cluster_post('/api/cluster/submit',
                                           _pipeline_build_train_spec(m, tf))
        if http_status == 200 and body.get('ok') and body.get('task_id'):
            train_tids[(m, tf)] = body['task_id']
        else:
            submit_failures.append(f'{m}@{tf}: {str(body)[:120]}')

    if not train_tids:
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": False,
               "error": f'no train submits succeeded — first 3 failures: {submit_failures[:3]}',
               "elapsed_s": round(time.time() - started, 1)}
        _emit_event("train", "cluster dispatch FAILED (no submits)",
                    error=out["error"])
        return out

    state = {
        'train_tids':  train_tids,
        'bt_tids':     {},
        'train_done':  set(),
        'bt_done':     set(),
        'cell_errors': {},
    }
    # Failed-submit cells are marked terminal so the loop exits cleanly
    for (m, tf) in cells:
        if (m, tf) not in train_tids:
            state['train_done'].add((m, tf))
            state['bt_done'].add((m, tf))
            state['cell_errors'][(m, tf)] = 'train submit failed at dispatch'

    def _submit_bt(m: str, tf: str, spec: dict) -> str | None:
        body, http_status = _cluster_post('/api/cluster/submit', spec)
        if http_status == 200 and body.get('ok') and body.get('task_id'):
            return body['task_id']
        return None

    POLL_S = 10.0
    DEADLINE_S = 12 * 3600
    last_status_emit = 0.0
    while True:
        time.sleep(POLL_S)
        if time.time() - started > DEADLINE_S:
            out = {"started_at": _now_iso(), "finished_at": _now_iso(),
                   "ok": False,
                   "error": f"pipeline deadline exceeded ({DEADLINE_S}s)",
                   "elapsed_s": round(time.time() - started, 1)}
            _emit_event("train", "DEADLINE EXCEEDED", error=out["error"])
            return out
        body, http_status = _cluster_get('/api/cluster/tasks')
        if http_status != 200 or not isinstance(body, list):
            continue
        by_id = {t.get('task_id'): t for t in body if isinstance(t, dict)}
        step = _pipeline_step(cells, state, by_id, _submit_bt)
        # Emit status update every ~30s and at the very end
        now = time.time()
        if step['finished'] or (now - last_status_emit) > 30:
            _update_status(train={
                "started_at":      None,   # preserve in patch merge
                "ok":              not state['cell_errors'],
                "finished_at":     _now_iso() if step['finished'] else None,
                "error":           None if not state['cell_errors'] else (
                                    f"{len(state['cell_errors'])} cell(s) failed"),
                "cells_total":     len(cells),
                "cells_complete":  step['cells_complete'],
                "train_inflight":  step['train_inflight'],
                "bt_inflight":     step['bt_inflight'],
            })
            last_status_emit = now
        if step['finished']:
            ok = not state['cell_errors']
            out = {"started_at": _now_iso(), "finished_at": _now_iso(),
                   "ok": ok,
                   "error": None if ok else (
                       f"{len(state['cell_errors'])} cell(s) failed — "
                       f"first 3: " +
                       "; ".join(f'{m}@{tf}: {e}'
                                  for (m, tf), e in list(state['cell_errors'].items())[:3])
                   ),
                   "cells_total":    len(cells),
                   "cells_complete": step['cells_complete'],
                   "elapsed_s":      round(time.time() - started, 1)}
            _emit_event("train",
                        f"cluster dispatch finished — {step['cells_complete']}/{len(cells)} cells",
                        ok=ok, elapsed_s=out["elapsed_s"])
            return out


def _run_train_phase() -> dict:
    """Phase 1: full multi-TF training. Cluster-routed by default
    (Phase 100e). AI_TRADER_PIPELINE_LOCAL=1 env var forces the legacy
    in-process train_all() path for emergency rollback.

    Returns {ok, error, ...} same shape regardless of route.
    """
    if _os.getenv('AI_TRADER_PIPELINE_LOCAL', '0') == '1':
        return _run_train_phase_local()
    return _run_train_phase_cluster()


def _run_train_phase_local() -> dict:
    """Legacy in-process train_all() path. Used only when
    AI_TRADER_PIPELINE_LOCAL=1 (emergency rollback). Workers stay idle —
    everything runs on the dashboard host.
    """
    started = time.time()
    _emit_event("train", "starting train_all (multi-TF, LOCAL)")
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


def _refresh_tf_pinning() -> int:
    """Post-backtest hook (Phase A). Reads the freshly-written
    latest_comparison + wf_results, computes best_tf per strategy via the
    same logic as /api/strategy/stability, and persists into
    data/strategy_tf_pinning.json. Returns the number of strategies pinned.
    """
    try:
        import json as _json
        import re as _re
        from src.engine import strategy_registry as _sr
        from src.engine.strategy_tf_pinning import update_auto_pins
        bt_path = PROJECT_ROOT / "data" / "backtest" / "latest_comparison.json"
        wf_path = PROJECT_ROOT / "data" / "backtest" / "wf_results.json"
        bt_rows = _json.loads(bt_path.read_text()) if bt_path.exists() else []
        wf_rows = _json.loads(wf_path.read_text()) if wf_path.exists() else []
        label_to_key = {info.get("label", nm): nm for nm, info in _sr.REGISTRY.items()}
        cells: dict[str, dict[str, dict]] = {}

        def _f(v):
            try: return float(v)
            except (TypeError, ValueError): return None

        for r in bt_rows:
            raw = (r.get("strategy") or "").strip()
            clean = _re.sub(r"^[AB]_", "", raw)
            key = label_to_key.get(clean, clean)
            tf = (r.get("timeframe") or "1h").strip() or "1h"
            b = cells.setdefault(key, {}).setdefault(tf, {"sharpe_sum": 0.0, "sharpe_n": 0,
                                                           "wf_sharpe_sum": 0.0, "wf_sharpe_n": 0})
            v = _f(r.get("sharpe"))
            if v is not None: b["sharpe_sum"] += v; b["sharpe_n"] += 1
        for r in wf_rows:
            key = (r.get("strategy") or "").strip()
            tf = (r.get("timeframe") or "1h").strip() or "1h"
            b = cells.get(key, {}).get(tf)
            if not b: continue
            v = _f(r.get("wf_mean_sharpe"))
            if v is not None: b["wf_sharpe_sum"] += v; b["wf_sharpe_n"] += 1

        best_tf: dict[str, str] = {}
        for strat, by_tf in cells.items():
            ranked = []
            for tf, b in by_tf.items():
                wf = b["wf_sharpe_sum"]/b["wf_sharpe_n"] if b["wf_sharpe_n"] else None
                sh = b["sharpe_sum"]/b["sharpe_n"]       if b["sharpe_n"]    else None
                score = wf if wf is not None else sh
                if score is not None: ranked.append((score, tf))
            if ranked:
                ranked.sort(reverse=True)
                best_tf[strat] = ranked[0][1]
        update_auto_pins(best_tf)
        return len(best_tf)
    except Exception as exc:
        logger.warning("refresh_tf_pinning failed: %s", exc)
        return 0


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
        # Phase A — refresh strategy_tf_pinning.json from the new results
        # so the bot's next signal cycle uses the freshly-computed best TFs.
        n_pins = _refresh_tf_pinning()
        out = {"started_at": _now_iso(), "finished_at": _now_iso(),
               "ok": True, "error": None, "rows_written": rows,
               "timeframes": list(timeframes),
               "elapsed_s": round(time.time() - started, 1),
               "tf_pins_written": n_pins}
        _emit_event("backtest",
                    f"run_full_backtest completed ({rows} rows, {n_pins} TF pins)",
                    rows=rows, n_pins=n_pins, elapsed_s=out["elapsed_s"])
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
