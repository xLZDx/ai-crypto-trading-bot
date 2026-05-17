"""Submit the full Stage-1 retrain to the cluster with worker pinning.

Pinning rules (per operator 2026-05-16):
  * model_type == "tft"  -> pin to RAZER          (16 GB VRAM)
  * model_type == "oft"  -> pin to WORKER-1-GPU   (Ivan, exclusive)
  * everything else      -> no pin (only one CPU lane = WORKER-1-CPU)

The orchestrator's `worker_name_pin` field (added in this session) restricts
dispatch to a worker whose `.name` matches. CPU models stay unpinned so the
scheduler can fan them out if more CPU lanes register later.

The losers (trend__1w, base__1d, futures__5m, futures__1m) were already
removed from training_rules.json's applicable_tfs in Stage 0b, so this
script won't emit tasks for them.

Symbol scope: full 20-symbol watchlist.

For TFT specifically: we want multi-symbol training (shared Darts model
across all watchlist symbols, not 20 separate models). So we submit ONE
TFT task per TF with symbol="ALL" — train_tft_model + the worker-side
ALL-sentinel handler will iterate `load_symbols()` internally.

OFT trains per (symbol, TF) cell — that's the existing OFT convention.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path("d:/test 2/AI trading assistance")
ORCH_URL = "http://127.0.0.1:7700"
KEY = "AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9"
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

TFT_PIN  = "RAZER"
OFT_PIN  = "WORKER-1-GPU"

# Load rules + watchlist
rules     = json.loads((ROOT / "data/training_rules.json").read_text(encoding="utf-8"))
watchlist = json.loads((ROOT / "data/watchlist.json").read_text(encoding="utf-8"))
print(f"watchlist size: {len(watchlist)}")
print(f"models in rules: {list((rules.get('models') or {}).keys())}")


def submit(spec: dict) -> str | None:
    try:
        r = requests.post(f"{ORCH_URL}/api/cluster/submit", headers=HEADERS,
                          data=json.dumps(spec), timeout=10)
        if r.status_code != 200:
            print(f"  submit FAIL status={r.status_code} body={r.text[:200]}")
            return None
        return (r.json() or {}).get("task_id")
    except Exception as e:
        print(f"  submit exc: {e}")
        return None


submitted: list[tuple[str, str, str, str]] = []  # (task_id, model, symbol, tf)

# --- 1) OFT first (Ivan GPU) -------------------------------------------------
oft_def = (rules.get("models") or {}).get("oft", {})
oft_tfs = oft_def.get("applicable_tfs") or []
if oft_tfs:
    for sym in watchlist:
        for tf in oft_tfs:
            spec = {
                "model_type": "oft",
                "symbol":     sym,
                "timeframe":  tf,
                "config": {
                    "epochs":           5,
                    "worker_name_pin":  OFT_PIN,
                    "sweep_id":         "full_retrain_2026-05-16",
                },
                "data_path":   str(ROOT / "data" / "raw" / f"{sym.replace('/','_')}_{tf}.csv.gz"),
                "output_path": str(ROOT / "models"),
            }
            tid = submit(spec)
            if tid:
                submitted.append((tid, "oft", sym, tf))

# --- 2) CPU models (no pin -- Ivan CPU is the only consumer) -----------------
CPU_MODELS = ["base", "trend", "futures", "scalping", "meta", "regime"]
for m_key in CPU_MODELS:
    m_def = (rules.get("models") or {}).get(m_key, {})
    tfs = m_def.get("applicable_tfs") or []
    if not tfs:
        continue
    # Map rules' model key to orchestrator's expected model_type
    mt = {
        "base":     "btc_rf",
        "trend":    "trend",
        "futures":  "futures_short",
        "scalping": "scalping",
        "meta":     "meta_labeler",
        "regime":   "regime",
    }.get(m_key, m_key)
    for sym in watchlist:
        for tf in tfs:
            spec = {
                "model_type": mt,
                "symbol":     sym,
                "timeframe":  tf,
                "config": {
                    "sweep_id": "full_retrain_2026-05-16",
                },
                "data_path":   str(ROOT / "data" / "raw" / f"{sym.replace('/','_')}_{tf}.csv.gz"),
                "output_path": str(ROOT / "models"),
            }
            tid = submit(spec)
            if tid:
                submitted.append((tid, mt, sym, tf))

# --- 3) TFT last (Razer, single multi-symbol task per TF, ALL sentinel) ------
tft_def = (rules.get("models") or {}).get("tft", {})
tft_tfs = tft_def.get("applicable_tfs") or []
tft_params = tft_def.get("params") or {}
for tf in tft_tfs:
    spec = {
        "model_type": "tft",
        "symbol":     "ALL",            # sentinel -> load_symbols() inside trainer
        "timeframe":  tf,
        "config": {
            "n_epochs":            int(tft_params.get("n_epochs", 3)),
            "min_epochs":          int(tft_params.get("min_epochs", 3)),
            "patience":            int(tft_params.get("patience", 5)),
            "input_chunk_length":  int(tft_params.get("input_chunk_length", 168)),
            "output_chunk_length": int(tft_params.get("output_chunk_length", 24)),
            "worker_name_pin":     TFT_PIN,
            "use_master_trainer":  True,
            "sweep_id":            "full_retrain_2026-05-16",
        },
    }
    tid = submit(spec)
    if tid:
        submitted.append((tid, "tft", "ALL", tf))

# Summary
print()
print(f"=== TOTAL SUBMITTED: {len(submitted)} ===")
from collections import Counter
by_model = Counter(m for _, m, _, _ in submitted)
for m, c in by_model.most_common():
    print(f"  {m:<14} {c}")
print()
print("Sweep ID: full_retrain_2026-05-16")
