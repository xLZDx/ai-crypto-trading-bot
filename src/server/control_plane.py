"""FastAPI control plane on :8100.

A read-mostly control surface that the dashboard at :5000 probes for health.
Exposes:

    GET  /health              - liveness probe (used by /api/monitor/services)
    GET  /status              - PIDs and alive state for bot/dash/training
    GET  /metrics             - balance, recent trade count, regime
    POST /control/bot/start   - launch_bot.ps1
    POST /control/bot/stop    - kill bot PID
    POST /control/training/start

Auth: requests to /control/* require X-API-Key header matching $env:API_KEY
(or DASHBOARD_API_KEY). /health, /status, /metrics are public.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PID_FILE     = PROJECT_ROOT / "data" / "process_ids.json"
STARTED_AT   = time.time()
VERSION      = "1.0"

app = FastAPI(
    title="AI Trader Control Plane",
    version=VERSION,
    description="Institutional control surface — health, PIDs, and bot lifecycle.",
)


def _read_pids() -> dict:
    if not PID_FILE.exists():
        return {}
    try:
        return json.loads(PID_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.status() not in ("zombie", "dead")
    except Exception:
        return False


def _require_api_key(x_api_key: str | None) -> None:
    expected = os.getenv("API_KEY") or os.getenv("DASHBOARD_API_KEY")
    if not expected:
        return  # no key configured -> open (dev mode)
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": VERSION,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "pid": os.getpid(),
    }


@app.get("/status")
def status() -> dict:
    pids = _read_pids()
    out = {}
    for key in ("bot", "dash", "monitor", "training", "realtime", "orch", "watchlist"):
        pid = pids.get(key)
        # PIDs in restart_all.ps1 are sometimes lists ("31852 40072")
        if isinstance(pid, str) and " " in pid:
            pid = pid.split()[0]
        out[key] = {"pid": pid, "alive": _alive(pid)}
    return {
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "components": out,
    }


@app.get("/metrics")
def metrics() -> dict:
    """Best-effort snapshot from the bot's persisted state."""
    out: dict = {}
    try:
        bal_path = PROJECT_ROOT / "data" / "balance_real.json"
        if bal_path.exists():
            out["balance_real"] = json.loads(bal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["balance_real_error"] = str(exc)
    try:
        bal_path = PROJECT_ROOT / "data" / "balance_virtual.json"
        if bal_path.exists():
            out["balance_virtual"] = json.loads(bal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["balance_virtual_error"] = str(exc)
    try:
        trades_path = PROJECT_ROOT / "data" / "trades.json"
        if trades_path.exists():
            data = json.loads(trades_path.read_text(encoding="utf-8"))
            trades = data if isinstance(data, list) else data.get("trades", [])
            out["trades_total"] = len(trades)
            cutoff = time.time() - 86400
            recent = [t for t in trades
                      if isinstance(t, dict) and float(t.get("timestamp", 0) or 0) > cutoff * 1000]
            out["trades_24h"] = len(recent)
    except Exception as exc:
        out["trades_error"] = str(exc)
    return out


@app.post("/control/bot/start")
def control_bot_start(x_api_key: str | None = Header(None)) -> dict:
    _require_api_key(x_api_key)
    pids = _read_pids()
    bot_pid = pids.get("bot")
    if isinstance(bot_pid, str) and " " in bot_pid:
        bot_pid = bot_pid.split()[0]
    if _alive(bot_pid):
        return {"started": False, "reason": "bot already running", "pid": bot_pid}
    script = PROJECT_ROOT / "launch_bot.ps1"
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"launcher missing: {script}")
    proc = subprocess.Popen(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=str(PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return {"started": True, "pid": proc.pid}


@app.post("/control/bot/stop")
def control_bot_stop(x_api_key: str | None = Header(None)) -> dict:
    _require_api_key(x_api_key)
    pids = _read_pids()
    bot_pid = pids.get("bot")
    if isinstance(bot_pid, str) and " " in bot_pid:
        bot_pid = bot_pid.split()[0]
    if not _alive(bot_pid):
        return {"stopped": False, "reason": "bot not running"}
    try:
        import psutil
        p = psutil.Process(int(bot_pid))
        p.terminate()
        try:
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            p.kill()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"stopped": True, "pid": bot_pid}


@app.post("/control/training/start")
def control_training_start(x_api_key: str | None = Header(None)) -> dict:
    _require_api_key(x_api_key)
    script = PROJECT_ROOT / "launch_training.ps1"
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"launcher missing: {script}")
    proc = subprocess.Popen(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=str(PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return {"started": True, "pid": proc.pid}


def main() -> int:
    import uvicorn
    host = os.getenv("FASTAPI_BIND_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_BIND_PORT", "8100"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
