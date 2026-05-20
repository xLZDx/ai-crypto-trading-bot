"""Wizard fix Phase A — apply asymmetric Triple Barrier (pt=4, sl=2).

Steps:
  1. Cancel ALL pending btc_rf + futures_short cells (sweep full_retrain_2026-05-16).
     Currently-running cells are LEFT alone — they'll finish on stale code,
     but the watchdog will see them complete and we accept that.
  2. Restart Ivan-CPU + Ivan-GPU workers so they pick up the new trainer code
     (Python sys.modules cached the old pt=2/sl=2; restart is the only clean
     way to force reload).
  3. Resubmit btc_rf + futures_short cells with worker_name_pins=[WORKER-1-CPU,
     WORKER-1-GPU]. RAZER is excluded because (a) it's mid-TFT and we don't
     want to interrupt, (b) it still has stale trainer code in memory.

Phase B (separate script, after Razer's TFT finishes naturally):
  - Cancel/resubmit pending TFT cells with dual-GPU pin
  - Restart Razer
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path("d:/test 2/AI trading assistance")
ORCH = "http://127.0.0.1:7700"
KEY  = "AC54481C0A1662C01AF95EF07E50038BAF239AC35899250B0CE3E605AFA220B9"
H    = {"X-API-Key": KEY, "Content-Type": "application/json"}

SWEEP_ID = "full_retrain_2026-05-16"
TARGET_MODEL_TYPES = {"btc_rf", "futures_short"}  # base + futures
IVAN_CPU = "http://192.168.0.167:7702"
IVAN_GPU = "http://192.168.0.167:7701"
IVAN_PINS = ["WORKER-1-CPU", "WORKER-1-GPU"]


def step1_cancel_pending():
    print("=== Step 1: cancel pending btc_rf + futures_short ===")
    state = json.loads((ROOT / "data/orchestrator_state.json").read_text(encoding="utf-8"))
    tasks = state.get("tasks") or {}
    targets = [t for t in tasks.values()
               if t.get("status") == "pending"
               and t.get("model_type") in TARGET_MODEL_TYPES
               and (t.get("config") or {}).get("sweep_id") == SWEEP_ID]
    print(f"  found {len(targets)} pending cells to cancel")
    cancelled = 0
    by_mt = {}
    for t in targets:
        tid = t["task_id"]
        mt  = t.get("model_type")
        try:
            r = requests.delete(f"{ORCH}/api/cluster/task/{tid}", headers=H, timeout=10)
            if r.status_code in (200, 204):
                cancelled += 1
                by_mt[mt] = by_mt.get(mt, 0) + 1
            else:
                print(f"  cancel FAIL {tid[:8]} status={r.status_code} body={r.text[:120]}")
        except Exception as e:
            print(f"  cancel EXC {tid[:8]}: {e}")
    print(f"  cancelled OK: {cancelled} | by_mt={by_mt}")
    return cancelled, by_mt


def step2_restart_workers():
    print("\n=== Step 2: restart Ivan workers ===")
    for label, url in (("WORKER-1-CPU", IVAN_CPU), ("WORKER-1-GPU", IVAN_GPU)):
        try:
            r = requests.post(f"{url}/restart", headers=H,
                              data=json.dumps({"confirm": True}), timeout=10)
            print(f"  {label} /restart -> {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"  {label} /restart EXC: {e}")
    # Wait for workers to come back
    print("  waiting 25s for workers to re-register...")
    time.sleep(25)


def step3_resubmit():
    print("\n=== Step 3: resubmit btc_rf + futures_short ===")
    rules     = json.loads((ROOT / "data/training_rules.json").read_text(encoding="utf-8"))
    watchlist = json.loads((ROOT / "data/watchlist.json").read_text(encoding="utf-8"))

    base_tfs    = ((rules.get("models") or {}).get("base") or {}).get("applicable_tfs") or []
    futures_tfs = ((rules.get("models") or {}).get("futures") or {}).get("applicable_tfs") or []
    print(f"  base TFs={base_tfs}  futures TFs={futures_tfs}  symbols={len(watchlist)}")

    submitted = 0
    by_mt = {}
    plan = [("btc_rf", base_tfs), ("futures_short", futures_tfs)]
    for mt, tfs in plan:
        if not tfs:
            print(f"  skip {mt} -- no applicable_tfs")
            continue
        for sym in watchlist:
            for tf in tfs:
                spec = {
                    "model_type": mt,
                    "symbol":     sym,
                    "timeframe":  tf,
                    "config": {
                        "sweep_id":         SWEEP_ID,
                        "worker_name_pins": IVAN_PINS,
                        "wizard_fix":       "asym_tb_pt4_sl2",
                    },
                    "data_path":   str(ROOT / "data" / "raw" / f"{sym.replace('/','_')}_{tf}.csv.gz"),
                    "output_path": str(ROOT / "models"),
                }
                try:
                    r = requests.post(f"{ORCH}/api/cluster/submit", headers=H,
                                      data=json.dumps(spec), timeout=10)
                    if r.status_code == 200:
                        submitted += 1
                        by_mt[mt] = by_mt.get(mt, 0) + 1
                    else:
                        print(f"  submit FAIL {mt} {sym} {tf} status={r.status_code} body={r.text[:120]}")
                except Exception as e:
                    print(f"  submit EXC {mt} {sym} {tf}: {e}")
    print(f"  submitted OK: {submitted} | by_mt={by_mt}")
    return submitted, by_mt


def main():
    print(f"Wizard fix Phase A -- sweep={SWEEP_ID}")
    print(f"Targets: {TARGET_MODEL_TYPES}, pins after resubmit: {IVAN_PINS}\n")

    n_cancelled, cancel_breakdown = step1_cancel_pending()
    step2_restart_workers()
    n_submitted, submit_breakdown = step3_resubmit()

    print("\n=== Phase A summary ===")
    print(f"  cancelled: {n_cancelled} {cancel_breakdown}")
    print(f"  submitted: {n_submitted} {submit_breakdown}")
    if n_cancelled and n_submitted >= n_cancelled - 5:
        print("  STATUS: OK (resubmit count >= cancelled - tolerance)")
        return 0
    else:
        print("  STATUS: PARTIAL -- review delta manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())
