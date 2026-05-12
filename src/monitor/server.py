"""Lightweight process monitor — http://localhost:5001
Shows live status, CPU/RAM, and last 100 log lines for every component.
Auto-refreshes every 5 seconds.
"""
import json
import html as _html
from pathlib import Path
from datetime import datetime

from flask import Flask, Response

app = Flask(__name__)

ROOT    = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
PID_FILE = ROOT / "data" / "process_ids.json"

COMPONENTS = [
    {"key": "dash",     "label": "Dashboard",    "log": "dashboard.log", "url": "http://127.0.0.1:5000"},
    {"key": "bot",      "label": "Trading Bot",  "log": "bot.log",       "url": None},
    {"key": "training", "label": "ML Training",  "log": "training.log",  "url": None},
    {"key": "download", "label": "Downloader",   "log": "download.log",  "url": None},
]


def _read_pids() -> dict:
    try:
        return json.loads(PID_FILE.read_text()) if PID_FILE.exists() else {}
    except Exception:
        return {}


def _proc_status(pid):
    if not pid:
        return "stopped", "-", "-"
    try:
        import psutil
        p = psutil.Process(int(pid))
        cpu = f"{p.cpu_percent(interval=0.05):.1f}%"
        mem = f"{p.memory_info().rss // (1024 * 1024)} MB"
        return "running", cpu, mem
    except Exception:
        return "stopped", "-", "-"


def _tail(log_name: str, n: int = 100) -> str:
    path = LOG_DIR / log_name
    if not path.exists():
        return "(no log yet)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return _html.escape("\n".join(lines[-n:]))
    except Exception:
        return "(error reading log)"


def _render() -> str:
    pids = _read_pids()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards = ""
    for c in COMPONENTS:
        pid    = pids.get(c["key"])
        status, cpu, mem = _proc_status(pid)
        dot    = "🟢" if status == "running" else "🔴"
        link   = f'<a href="{c["url"]}" target="_blank" style="color:#58a6ff">{c["url"]}</a> &nbsp;' if c["url"] else ""
        meta   = f'PID {pid} &nbsp;|&nbsp; CPU {cpu} &nbsp;|&nbsp; RAM {mem}' if pid else "not started"
        log_content = _tail(c["log"])

        cards += f"""
<div class="card">
  <div class="card-header">
    <span class="dot">{dot}</span>
    <b>{c['label']}</b>
    <span class="meta">&nbsp;{meta}</span>
    <span style="float:right">{link}</span>
  </div>
  <pre class="log" id="log-{c['key']}">{log_content}</pre>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>AI Trader Monitor</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Cascadia Code', 'Consolas', monospace;
          background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1   {{ color: #58a6ff; margin-bottom: 4px; }}
  .ts  {{ color: #8b949e; font-size: 12px; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
  .card-header {{ padding: 12px 16px; border-bottom: 1px solid #21262d;
                  display: flex; align-items: center; gap: 6px; }}
  .meta  {{ color: #8b949e; font-size: 12px; }}
  .log   {{ background: #010409; padding: 12px 16px; font-size: 11px;
             max-height: 320px; overflow-y: auto; white-space: pre-wrap;
             word-break: break-all; color: #e6edf3; line-height: 1.45; }}
</style>
</head>
<body>
<h1>AI Trader Monitor</h1>
<div class="ts">Last updated: {now} &nbsp;|&nbsp; Auto-refresh every 5 s</div>
{cards}
</body>
</html>"""


@app.route("/")
def index():
    return Response(_render(), mimetype="text/html")


@app.route("/logs/<component>")
def raw_log(component: str):
    safe = {c["key"]: c["log"] for c in COMPONENTS}
    log_file = safe.get(component)
    if not log_file:
        return Response("unknown component", status=404)
    path = LOG_DIR / log_file
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "(no log yet)"
    return Response(text, mimetype="text/plain")


if __name__ == "__main__":
    import os
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 2026-05-12 Phase A2: bind to localhost by default. The monitor UI
    # is consumed by the dashboard's same-host JS only, so 127.0.0.1 is
    # the right default. Override via MONITOR_BIND_HOST env var if you
    # ever need cross-machine access (and add auth first — these
    # endpoints are currently unauthenticated).
    bind_host = os.getenv("MONITOR_BIND_HOST", "127.0.0.1")
    print(f"Monitor running at http://{bind_host}:5001")
    app.run(host=bind_host, port=5001, debug=False, use_reloader=False)
