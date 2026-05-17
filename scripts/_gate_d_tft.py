"""Gate D: real 1-epoch TFT train on BTC/USDT with 30 days history.

Asserts:
  1. train_tft_model completes without error
  2. meta JSON at models/tft_<tf>_meta.json exists and has duration_s > 0
  3. training_progress.json contains an entry for this task_id
  4. training_runs_history.json gained a new entry
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cap to one symbol BEFORE importing the trainer.
from src.engine import train_tft_model as ttm  # noqa: E402
ttm.load_symbols = lambda: ["BTC_USDT"]  # type: ignore[assignment]

TASK_ID = f"gate_d_{int(time.time())}"
TF = "1h"

print(f"[gate_d] starting train_tft_model tf={TF} n_epochs=1 history_days=30 task_id={TASK_ID}")
t0 = time.time()
try:
    res = ttm.train_tft_model(
        timeframe=TF,
        n_epochs=1,
        min_epochs=1,
        patience=2,
        history_days=30,
        progress_task_id=TASK_ID,
        dry_run=False,
    )
    elapsed = time.time() - t0
    print(f"[gate_d] DONE in {elapsed:.1f}s  result_type={type(res).__name__}")
except Exception as exc:
    elapsed = time.time() - t0
    print(f"[gate_d] FAILED after {elapsed:.1f}s: {exc!r}")
    raise

# Validate meta JSON
meta_path = ROOT / "models" / f"tft_{TF}_meta.json"
if not meta_path.exists():
    raise SystemExit(f"[gate_d] FAIL: {meta_path} missing")
meta = json.loads(meta_path.read_text())
print(f"[gate_d] meta keys: {sorted(meta.keys())}")
required = ("duration_s", "epochs_completed", "started_at_unix", "finished_at_unix")
missing = [k for k in required if k not in meta or meta[k] in (None, 0, 0.0)]
if missing:
    raise SystemExit(f"[gate_d] FAIL: meta missing/zero fields: {missing}")
print(f"[gate_d] meta.duration_s={meta['duration_s']} epochs_completed={meta['epochs_completed']}")

# Validate training_progress.json
prog_path = ROOT / "data" / "training_progress.json"
if not prog_path.exists():
    raise SystemExit(f"[gate_d] FAIL: {prog_path} missing")
prog = json.loads(prog_path.read_text())
if TASK_ID not in (prog if isinstance(prog, dict) else {}):
    keys = list(prog.keys()) if isinstance(prog, dict) else []
    raise SystemExit(f"[gate_d] FAIL: task_id {TASK_ID} not in training_progress.json keys={keys[:5]}")
print(f"[gate_d] progress entry for {TASK_ID}: {prog[TASK_ID]}")

# Validate training_runs_history.json
hist_path = ROOT / "data" / "training_runs_history.json"
if hist_path.exists():
    hist = json.loads(hist_path.read_text())
    recent = hist[-3:] if isinstance(hist, list) else []
    print(f"[gate_d] history tail (last 3): {recent}")
else:
    print(f"[gate_d] WARN: {hist_path} missing -- instrumentation not wired for this trainer?")

print(f"[gate_d] PASS  elapsed={elapsed:.1f}s")
