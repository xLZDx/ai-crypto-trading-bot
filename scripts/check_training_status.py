"""Local-only training status inspector.

Reads the local TFT training log (UTF-16 LE), the model file metadata, and
the live dashboard's `/api/strategy/full` endpoint to produce a single JSON
report at `data/training_status_report.json`. Designed to be invoked by
Windows Task Scheduler on a recurring cadence — NO external network calls.

Usage:
    python scripts/check_training_status.py
    python scripts/check_training_status.py --quiet     # no stdout, only file
    python scripts/check_training_status.py --json      # stdout = JSON only

Exit codes:
    0  report written successfully (regardless of training state)
    1  unrecoverable error (e.g. project root unreadable)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "logs" / "tft_3epoch.log"
MODEL_PATH   = PROJECT_ROOT / "models" / "tft_model.pt"
META_PATH    = PROJECT_ROOT / "models" / "tft_model_meta.json"
CKPT_PATH    = PROJECT_ROOT / "models" / "tft_model.pt.ckpt"
REPORT_PATH  = PROJECT_ROOT / "data" / "training_status_report.json"
DASHBOARD    = "http://127.0.0.1:5000"


def _decode_log() -> str:
    """The TFT log is UTF-16 LE because PowerShell's Tee-Object writes that
    by default. Decode robustly and fall back to UTF-8 if needed."""
    if not LOG_PATH.exists():
        return ""
    raw = LOG_PATH.read_bytes()
    for enc in ("utf-16", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_EPOCH_LINE_RE = re.compile(r"Epoch (\d+):\s*(\d+)%.*?(\d+)/(\d+).*?train_loss=([0-9.]+)(?:.*?val_loss=([0-9.]+))?")


def _parse_log() -> dict:
    txt = _decode_log()
    out: dict = {
        "log_present": bool(txt),
        "log_path": str(LOG_PATH),
        "log_size_bytes": LOG_PATH.stat().st_size if LOG_PATH.exists() else 0,
        "lines_total": 0,
        "epochs_observed": 0,
        "epochs_completed": 0,
        "current_epoch": None,
        "current_progress_pct": None,
        "last_train_loss": None,
        "last_val_loss": None,
        "last_line_excerpt": "",
    }
    if not txt:
        return out
    lines = txt.splitlines()
    out["lines_total"] = len(lines)
    if lines:
        out["last_line_excerpt"] = lines[-1][:200]

    epoch_pcts: dict[int, int] = {}
    last_train = last_val = None
    cur_epoch = cur_pct = None
    for line in lines:
        m = _EPOCH_LINE_RE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        pct = int(m.group(2))
        epoch_pcts[ep] = max(epoch_pcts.get(ep, 0), pct)
        cur_epoch = ep
        cur_pct = pct
        last_train = float(m.group(5))
        if m.group(6):
            last_val = float(m.group(6))

    out["epochs_observed"] = len(epoch_pcts)
    out["epochs_completed"] = sum(1 for p in epoch_pcts.values() if p >= 100)
    out["current_epoch"] = cur_epoch
    out["current_progress_pct"] = cur_pct
    out["last_train_loss"] = last_train
    out["last_val_loss"] = last_val
    return out


def _stat_path(p: Path) -> dict:
    if not p.exists():
        return {"present": False, "path": str(p)}
    st = p.stat()
    return {
        "present": True,
        "path": str(p),
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "age_s": round(time.time() - st.st_mtime, 1),
    }


def _read_meta() -> dict:
    if not META_PATH.exists():
        return {"present": False}
    try:
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        data["present"] = True
        return data
    except Exception as exc:
        return {"present": False, "error": str(exc)}


def _check_dashboard() -> dict:
    """Hit the LOCAL Flask dashboard. Loopback only — no external comms."""
    out = {"reachable": False}
    try:
        req = urllib.request.Request(DASHBOARD + "/api/models")
        with urllib.request.urlopen(req, timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        tft = data.get("tft", {}) if isinstance(data, dict) else {}
        out.update({
            "reachable": True,
            "tft_in_response": bool(tft),
            "tft_last_trained": tft.get("last_trained"),
            "tft_model_path": tft.get("model_path"),
            "tft_n_epochs": tft.get("n_epochs"),
        })
    except urllib.error.URLError as exc:
        out["error"] = f"dashboard_unreachable: {exc.reason}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_report() -> dict:
    log_info   = _parse_log()
    model_info = _stat_path(MODEL_PATH)
    ckpt_info  = _stat_path(CKPT_PATH)
    meta       = _read_meta()
    dash       = _check_dashboard()

    target_epochs = int(meta.get("n_epochs", 3)) if meta.get("present") else 3
    completed = log_info["epochs_completed"]
    is_done = (
        model_info["present"]
        and meta.get("present", False)
        and completed >= target_epochs
    )
    if is_done:
        status = "completed"
    elif completed > 0 or log_info["current_epoch"] is not None:
        status = "in_progress"
    else:
        status = "not_started"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "local",
        "execution": "LOCAL_ONLY",
        "status": status,
        "epochs_target": target_epochs,
        "epochs_completed": completed,
        "log": log_info,
        "model": model_info,
        "checkpoint": ckpt_info,
        "meta": meta,
        "dashboard": dash,
        "summary_bullets": _bullets(log_info, model_info, meta, dash, status),
    }
    return report


def _bullets(log_info, model_info, meta, dash, status) -> list[str]:
    bullets = []
    if status == "completed":
        bullets.append(f"OK — TFT 3-epoch training COMPLETED. {log_info['epochs_completed']} epoch(s) finished.")
    elif status == "in_progress":
        bullets.append(f"IN PROGRESS — at epoch {log_info['current_epoch']} ({log_info['current_progress_pct']}%).")
    else:
        bullets.append("NOT STARTED — no log activity detected.")

    if log_info["last_train_loss"] is not None:
        bullets.append(
            f"Final losses: train={log_info['last_train_loss']:.4f}"
            + (f", val={log_info['last_val_loss']:.4f}" if log_info['last_val_loss'] is not None else "")
        )
    if model_info["present"]:
        bullets.append(
            f"Model file present: {model_info['size_bytes']/1e6:.1f} MB, mtime {model_info['mtime_iso']}"
        )
    else:
        bullets.append("Model file NOT present at " + str(MODEL_PATH))

    if dash.get("reachable"):
        if dash.get("tft_in_response"):
            bullets.append(f"Dashboard reflects TFT (last_trained={dash.get('tft_last_trained')}).")
        else:
            bullets.append("Dashboard reachable but TFT entry missing in /api/models.")
    else:
        bullets.append(f"Dashboard NOT reachable: {dash.get('error', 'unknown')}.")

    return bullets


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true",
                   help="suppress stdout (file only)")
    p.add_argument("--json", action="store_true",
                   help="print full JSON report to stdout instead of bullets")
    args = p.parse_args()

    try:
        report = build_report()
    except Exception as exc:
        print(f"ERROR building report: {exc}", file=sys.stderr)
        return 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    elif not args.quiet:
        print(f"[check_training_status] {report['status'].upper()}")
        for b in report["summary_bullets"]:
            print(f"  - {b}")
        print(f"  - report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
