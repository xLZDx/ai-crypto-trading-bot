"""Error monitor — tails recent log activity AND probes live status surfaces,
classifies, deduplicates.

Used by /api/errors/recent and the dashboard banner. The goal is to catch
the user's attention when something is *actively* breaking (last 30 min)
without spamming once an issue stops recurring.

Design:
  - LOG path: tail the last ~500 lines of each watched log file (no
    streaming; polled on every /api/errors/recent request, plus a 30s
    background refresh so the cache stays warm). Classify each line via
    simple regex into severity {critical, warning}. Strip numeric/timestamp/
    symbol tokens to a "signature" so e.g. all 17 "Low confidence (0.4x)
    — no trade" collapse to a single entry with count=N.
  - SURFACE path: poll the same status surfaces the dashboard exposes —
    monitor/services probes, agent_status.json, cluster orchestrator,
    scheduler report — and inject any non-OK state as a banner entry.
    Surface entries auto-heal as soon as the probe flips back to OK
    (no need to wait for AUTO_CLEAR_S).
  - Entries carry a `source` field: 'log' or 'surface'. Log entries
    auto-clear after AUTO_CLEAR_S of silence; surface entries clear as
    soon as the probe returns OK. State persists to data/error_state.json
    across restarts so a freshly-rebooted dashboard still shows what just
    happened.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR     = PROJECT_ROOT / "logs"
STATE_FILE   = PROJECT_ROOT / "data" / "error_state.json"

# Files we tail. Order matters only for tie-breaking on "first_seen".
WATCHED_FILES = (
    "bot.log",
    "dashboard.log",
    "orderbook_collector.log",
    "realtime_db.log",
    "fastapi.log",
    "data_orchestrator.log",
)

TAIL_LINES        = 500
TAIL_BYTES        = 200_000        # cap per file to avoid runaway reads
AUTO_CLEAR_S      = 30 * 60        # 30 min — nothing seen → entry drops off
ACTIVE_WINDOW_S   = 30 * 60        # banner shows entries with last_seen ≤ this
REFRESH_S         = 30.0           # background refresh cadence
MAX_ENTRIES       = 50             # state cap
# Cool-off period after CLEAR / dismiss(key). While the deadline for a key
# is in the future, scan() / scan_status_surfaces() will not re-create it.
# Without this, "Clear All" was a visual no-op against any still-firing
# issue: the very next /api/errors/recent poll re-ran scan(), re-matched
# the same log lines (which are <30 min old), and re-added the entries.
DISMISS_SUPPRESS_S = 5 * 60        # 5 min cool-off after CLEAR / dismiss

# Patterns. We classify on the LOG LEVEL — the all-caps token between
# timestamp and message body in the standard logging format
# `2026-05-04 21:30:00,123 - LEVEL - body...`. This avoids treating a
# `WARNING - Error loading news...` line as critical just because the
# body happens to contain the word "error".
#
# WARNING comes BEFORE ERROR/CRITICAL in the alternation so we don't fall
# through to the broader pattern when both could match.
_LEVEL_RE = re.compile(
    r"(?:^|\s)-\s+(WARNING|ERROR|CRITICAL|FATAL)\s+-\s",
)
# Free-form patterns for log lines without the standard level prefix.
# We require a real exception type token (e.g. "ValueError:", "RuntimeError:")
# rather than the bare "Traceback" header — a lone "Traceback (most recent
# call last):" line carries no actionable info on its own and was firing as
# a separate critical entry every time it appeared near the tail boundary.
_TRACEBACK_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z]+Error|FatalError|SystemExit|KeyboardInterrupt):\s",
)

# Lines that look like errors but are routine and should NOT alarm the user.
# (Free-tier 429 spam, expected reconnects, etc.)
_BENIGN_RE = re.compile(
    r"|".join((
        # Gemini free-tier rate-limit cascade (handled by cooldown logic)
        r"429 RESOURCE_EXHAUSTED",
        r"429 Too Many Requests",
        r"AFC is enabled",
        # Expected websocket churn
        r"WebSocket disconnected.*Reconnecting",
        # GARCH normal operation (it's an indicator, not an error)
        r"GARCH predicts sharp volatility",
        r"GARCH spike! Position halved",
        # Informational signal-pipeline messages
        r"Meta-Labeler missing required",
        r"Meta-Labeler scoring with neutral priors",
        r"Low confidence \(",
        r"is not strictly mean-reversion",
        r"do not strictly exhibit mean-reversion",
        r"OFT vetoed",
        r"OFT BLOCK",
        # By-design: spot-only assets have no perp futures
        r"No perpetual futures market for ",
        # Optional Darts dependencies — we don't need StatsForecast or XGBoost
        r"StatsForecast module could not be imported",
        r"XGBoost.*module could not be imported",
        # Flask dev-server banner (we run dev mode in this environment)
        r"This is a development server\. Do not use",
        # Transient external feed hiccups (news 502 / RSS 5xx / timeouts)
        r"Error loading news from .*HTTP Error 5\d\d",
        r"Error loading news from .*read operation timed out",
        r"Error loading news from .*timed out",
        r"HTTP Error 502: Bad Gateway",
        # Fix A success path: bot detects exchange-side position is already
        # closed (0 contracts) and force-closes locally on the first tick
        # instead of cascading through 3 retry attempts. This is the
        # DESIRED behavior, not a fault.
        r"Force-closed SCALPING position .* exchange has 0 contracts \(already closed\)",
        r"Force-closed SCALPING position .* Binance reduceOnly rejected",
        # Stale launch-attempt errors: the launcher tried to spawn a python
        # module before the working directory was set correctly, leaving
        # a one-line ModuleNotFoundError in the log. The current process
        # is fine; we just want this one stale tail-line to stop pinging.
        r"Error while finding module specification for 'src\.",
        r"No module named 'src'",
        r"'D:\\test' is not recognized as an internal or external command",
        # PowerShell error frames left over from the launch_*.ps1 Out-File
        # race bug (now fixed in commit 76d521b — but historical lines
        # sit in dashboard.log/fastapi.log/bot.log tails for a while).
        # These error frames have no embedded timestamp, so the
        # timestamp-skip in scan() can't drop them. BENIGN them.
        r"^\s*\+ CategoryInfo\s+:\s*OpenError",
        r"^\s*\+ FullyQualifiedErrorId\s*:\s*FileOpenFailure",
        r"Out-File : The process cannot access the file .* because it is being",
        r"\$_; \"\$_\" \| Out-File -FilePath",
        # Windows asyncio / FastAPI client-disconnect noise
        r"ConnectionResetError:.*WinError 10054",
        r"Exception in callback _ProactorBasePipeTransport",
        r"_call_connection_lost",
        # An orphan "Traceback" header by itself (no body context) — it gets
        # captured because we only read tail-of-file. The classifier no
        # longer routes these to "critical" via _TRACEBACK_RE; this line
        # in BENIGN catches any pre-existing pattern variants.
        r"^Traceback \(most recent call last\):\s*$",
    )),
    re.IGNORECASE | re.MULTILINE,
)

# Strip volatility from each line so "ETH/USDT" and "BTC/USDT" hashing
# to the same signature. Order: strip tokens that vary across instances.
_NORMALIZERS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2},?\d*\b"), "<TS>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2},\d+\b"), "<TS>"),
    (re.compile(r"\b[A-Z]{2,5}_USDT\b"), "<SYM>"),
    (re.compile(r"\b[A-Z]{2,5}/USDT(?::USDT)?\b"), "<SYM>"),
    (re.compile(r"\bPID\s+\d+\b"), "<PID>"),
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"-?\d+\.\d+e[+-]\d+"), "<NUM>"),
    (re.compile(r"-?\d+\.\d+"), "<NUM>"),
    (re.compile(r"\b\d{4,}\b"), "<NUM>"),
    (re.compile(r"\s+"), " "),
]

_state_lock = threading.Lock()
_state: dict[str, dict[str, Any]] = {}
# Suppression map: key -> epoch timestamp until which scans should skip
# re-creating this key. Lives alongside _state and is persisted with it.
_dismissed_until: dict[str, float] = {}
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()


def _is_suppressed(key: str, now: float) -> bool:
    """Return True iff this key is still under a dismiss cool-off. Expired
    deadlines are removed in-place so the map can't grow unbounded."""
    deadline = _dismissed_until.get(key)
    if deadline is None:
        return False
    if deadline > now:
        return True
    _dismissed_until.pop(key, None)
    return False


def _signature(line: str) -> str:
    s = line.strip()
    for pat, sub in _NORMALIZERS:
        s = pat.sub(sub, s)
    return s[:240]


# Signature prefix marking a surface-derived entry. Used so _load_state
# skips re-classification (the sample isn't a log line so _classify
# would always return None) and so surface entries can be located for
# auto-healing without scanning every entry.
_SURFACE_PREFIX = "surface:"


def _tail(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > TAIL_BYTES:
                f.seek(size - TAIL_BYTES)
                f.readline()  # discard partial line
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-TAIL_LINES:]
    except Exception as exc:
        logger.debug("[error_monitor] tail %s: %s", path, exc)
        return []


def _load_state() -> None:
    """Load persisted state, then re-validate every entry against the
    CURRENT classifier. Lines that used to be flagged but are now BENIGN
    (because the classifier has been updated) get dropped here so the UI
    doesn't keep showing stale alarms after a code update."""
    global _state, _dismissed_until
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        _state = {}
        _dismissed_until = {}
        return
    # Backwards-compat: older state files were a flat {key: entry} map.
    # New files wrap it as {"entries": {...}, "dismissed_until": {...}}.
    if isinstance(raw, dict) and "entries" in raw and isinstance(raw["entries"], dict):
        entries_raw = raw.get("entries") or {}
        dismissed_raw = raw.get("dismissed_until") or {}
    else:
        entries_raw = raw
        dismissed_raw = {}
    now = time.time()
    _dismissed_until = {
        k: float(v) for k, v in (dismissed_raw or {}).items()
        if isinstance(v, (int, float)) and float(v) > now
    }
    cleaned: dict[str, dict[str, Any]] = {}
    for key, entry in (entries_raw or {}).items():
        if not isinstance(entry, dict):
            continue
        # Surface entries are not log lines — _classify would always
        # return None for them. Trust their stored kind and let the next
        # scan_status_surfaces() decide whether to keep them.
        if entry.get("source") == "surface":
            cleaned[key] = entry
            continue
        sample = entry.get("sample", "")
        kind_now = _classify(sample)
        if kind_now is None:
            # Was flagged before but the BENIGN list / classifier rules
            # have changed and this line is no longer actionable.
            continue
        if kind_now != entry.get("kind"):
            # Severity bumped or downgraded — re-key under the new kind.
            entry = dict(entry)
            entry["kind"] = kind_now
            new_key = f"{kind_now}::{entry.get('file','')}::{entry.get('signature','')}"
            cleaned[new_key] = entry
        else:
            cleaned[key] = entry
    _state = cleaned


def _save_state() -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Drop any expired suppressions before persisting so the file
        # doesn't accumulate dead keys forever.
        now = time.time()
        live_dismissed = {k: v for k, v in _dismissed_until.items() if v > now}
        payload = {"entries": _state, "dismissed_until": live_dismissed}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.debug("[error_monitor] save state: %s", exc)


# Parse the line's own log timestamp (e.g. "2026-05-06 12:28:01,834") so we
# can skip lines whose own clock says they're older than the active window.
# Otherwise the same stale ERROR line in the file's tail keeps refreshing
# its last_seen on every scan and never auto-clears.
_LOG_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[,.]\d+)?"
)


def _line_age_s(line: str) -> float | None:
    """Return seconds since the line's embedded timestamp, or None if no
    parseable timestamp is present at the line start."""
    m = _LOG_TS_RE.match(line.strip())
    if not m:
        return None
    try:
        # Use space form for fromisoformat compatibility
        ts_str = m.group(1).replace("T", " ")
        dt = datetime.fromisoformat(ts_str)
        # Logs are in local time; compare against local-naive now()
        return (datetime.now() - dt).total_seconds()
    except Exception:
        return None


def _classify(line: str) -> str | None:
    if _BENIGN_RE.search(line):
        return None
    m = _LEVEL_RE.search(line)
    if m:
        lvl = m.group(1).upper()
        if lvl in ("ERROR", "CRITICAL", "FATAL"):
            return "critical"
        if lvl == "WARNING":
            return "warning"
    # Fallback for lines without "- LEVEL -" prefix (raw stderr, tracebacks).
    if _TRACEBACK_RE.search(line):
        return "critical"
    return None


def scan() -> dict[str, dict[str, Any]]:
    """One full pass over the watched logs. Updates the in-memory state and
    persists. Returns a SNAPSHOT (so callers can serialise without a lock)."""
    now = time.time()
    with _state_lock:
        # Read each file's tail and update entries.
        for fname in WATCHED_FILES:
            path = LOGS_DIR / fname
            for line in _tail(path):
                kind = _classify(line)
                if kind is None:
                    continue
                # Skip lines whose own timestamp says they're older than the
                # active window. Otherwise a stale ERROR sitting in the
                # tail keeps refreshing its last_seen on every scan and never
                # ages out — which is exactly what happened after the
                # restart_all detach iterations: old failed-launch errors
                # stuck in the log tail were re-flagged each scan despite
                # the underlying processes being healthy.
                age = _line_age_s(line)
                if age is not None and age > ACTIVE_WINDOW_S:
                    continue
                sig = _signature(line)
                key = f"{kind}::{fname}::{sig}"
                # CLEAR ALL / dismiss-key cool-off: while a key is under
                # suppression, scan() must not re-create or update it. The
                # underlying log line is still in the tail, so without this
                # the next poll resurrects the entry and the user sees no
                # effect from clicking CLEAR ALL.
                if _is_suppressed(key, now):
                    continue
                e = _state.get(key)
                if e is None:
                    if len(_state) >= MAX_ENTRIES:
                        # Drop the oldest entry by first_seen.
                        oldest = min(_state.items(),
                                     key=lambda kv: kv[1].get("first_seen", 0))
                        _state.pop(oldest[0], None)
                    e = {
                        "kind":       kind,
                        "file":       fname,
                        "signature":  sig,
                        "sample":     line.strip()[:300],
                        "source":     "log",
                        "first_seen": now,
                        "last_seen":  now,
                        "count":      1,
                    }
                    _state[key] = e
                else:
                    e["last_seen"] = now
                    e["count"]     = int(e.get("count", 0)) + 1
                    # Update sample to the latest line (so the user sees
                    # current values, not a stale numeric snapshot).
                    e["sample"]    = line.strip()[:300]

        # Auto-clear (LOG entries only): unseen for AUTO_CLEAR_S drop off.
        # Surface entries are managed by scan_status_surfaces() and
        # auto-heal as soon as the probe goes green — they intentionally
        # ignore last_seen here.
        stale = [k for k, e in _state.items()
                 if e.get("source", "log") == "log"
                 and (now - e.get("last_seen", 0)) > AUTO_CLEAR_S]
        for k in stale:
            _state.pop(k, None)

        snapshot = {k: dict(v) for k, v in _state.items()}

    _save_state()
    return snapshot


# ─── Status-surface probes ────────────────────────────────────────────────────
#
# Each probe returns either None (surface OK — no banner entry) or a tuple
# (kind, signature_suffix, sample_text) describing the fault. The signature
# carries the _SURFACE_PREFIX marker so _load_state can tell surface entries
# from log entries.
#
# NOTE: probes intentionally do not use HTTP to talk to localhost:5000 —
# they read the same JSON status files / call the same Python helpers the
# Flask routes use. This avoids the dashboard talking to itself.


def _probe_parquet_store() -> tuple[str, str, str] | None:
    """Probe the ParquetClient store — replaces the QuestDB HTTP probe.
    The store is healthy iff DuckDB imports + the data dir is writable +
    the singleton client reports available. No network port involved."""
    try:
        from src.database.parquet_client import get_client
        c = get_client()
        if c.is_available(force=True):
            return None
        return ("critical", "service:parquet_store",
                "ParquetClient store unavailable — DuckDB missing or "
                "data/db/ not writable")
    except Exception as exc:
        return ("critical", "service:parquet_store",
                f"ParquetClient init failed — {type(exc).__name__}: {exc}")


def _probe_duckdb() -> tuple[str, str, str] | None:
    try:
        import duckdb  # noqa: F401
        tmp = PROJECT_ROOT / "data" / "cache" / "duckdb_temp"
        tmp.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        try:
            con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
            con.execute("SELECT 1").fetchone()
        finally:
            con.close()
        return None
    except Exception as exc:
        return ("critical", "service:duckdb",
                f"DuckDB DOWN — {type(exc).__name__}: {exc}")


def _probe_parquet() -> tuple[str, str, str] | None:
    try:
        pq_root = PROJECT_ROOT / "data" / "parquet"
        if not pq_root.exists():
            return ("warning", "service:parquet",
                    "Parquet store missing (data/parquet/) — "
                    "run scripts/migrate_news_to_parquet.py / archive downloader")
        files = list(pq_root.rglob("*.parquet"))
        if not files:
            return ("warning", "service:parquet",
                    "Parquet store has 0 files — backfill not run yet")
        return None
    except Exception as exc:
        return ("warning", "service:parquet", f"Parquet probe error: {exc}")


def _probe_fastapi() -> tuple[str, str, str] | None:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=0.5) as resp:
            if resp.status == 200:
                return None
            return ("warning", "service:fastapi",
                    f"FastAPI control plane HTTP {resp.status} on :8100")
    except Exception as exc:
        return ("warning", "service:fastapi",
                f"FastAPI control plane DOWN on :8100 — {type(exc).__name__}")


def _probe_realtime() -> tuple[str, str, str] | None:
    try:
        rt_path = PROJECT_ROOT / "data" / "realtime_status.json"
        if not rt_path.exists():
            return ("warning", "service:realtime",
                    "Realtime feed: no status file (orderbook_realtime.py "
                    "hasn't written one yet)")
        st = json.loads(rt_path.read_text(encoding="utf-8"))
        if not bool(st.get("connected")):
            sym = st.get("symbol", "?")
            return ("warning", "service:realtime",
                    f"Realtime feed (Binance L2) DISCONNECTED · sym: {sym}")
        # Stale heartbeat: connected=true but last message > 60s old → still bad
        last = st.get("last_msg_ts") or st.get("last_msg")
        if isinstance(last, (int, float)) and (time.time() - float(last)) > 60:
            return ("warning", "service:realtime",
                    f"Realtime feed STALE — last message {int(time.time() - float(last))}s ago")
        return None
    except Exception as exc:
        return ("warning", "service:realtime", f"Realtime feed probe error: {exc}")


def _probe_processes() -> list[tuple[str, str, str]]:
    """Read data/process_ids.json and verify the bot process is alive.
    Dashboard process is implicit (we're running in it). Returns a list
    so a single probe can emit multiple faults."""
    faults: list[tuple[str, str, str]] = []
    try:
        pid_path = PROJECT_ROOT / "data" / "process_ids.json"
        if not pid_path.exists():
            return [("warning", "process:bot",
                     "Trading bot process not started yet (no data/process_ids.json)")]
        pids = json.loads(pid_path.read_text(encoding="utf-8") or "{}")
        bot_pid = pids.get("bot")
        if not bot_pid:
            faults.append(("critical", "process:bot",
                           "Trading bot PID missing — restart_all.ps1 didn't launch it"))
        else:
            try:
                import psutil  # type: ignore
                if not psutil.pid_exists(int(bot_pid)):
                    faults.append(("critical", "process:bot",
                                   f"Trading bot process {bot_pid} is DEAD — bot loop not running"))
            except ImportError:
                # psutil missing — skip the liveness check rather than false-alarm.
                pass
            except Exception:
                pass
    except Exception as exc:
        faults.append(("warning", "process:bot", f"Process probe error: {exc}"))
    return faults


_USER_INITIATED_AGENTS = frozenset({
    # These agents only tick while the user is actively running something
    # from the corresponding tab. When idle, they leave a stale "running"
    # heartbeat in agent_status.json — that's expected, not a fault.
    # All three are constructed in _do_sim_init() but only START when the
    # user clicks Start in the Simulator tab (_apply_sim_start in app.py).
    "SimulatorAgent",
    "StrategySimulatorAgent",
    "ContinuousTrainerAgent",
})


def _probe_agents() -> list[tuple[str, str, str]]:
    """Read data/agent_status.json and flag any agent in error state OR
    with a stale heartbeat (> 3× its own interval_sec). User-initiated
    agents (Simulator/StrategySimulator) are exempt from the staleness
    check — their heartbeat going quiet just means the user closed the sim."""
    faults: list[tuple[str, str, str]] = []
    try:
        agent_path = PROJECT_ROOT / "data" / "agent_status.json"
        if not agent_path.exists():
            return faults
        live = json.loads(agent_path.read_text(encoding="utf-8") or "{}")
        now = time.time()
        for name, entry in (live or {}).items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).lower()
            if status in ("error", "crashed", "failed", "fault"):
                task = entry.get("current_task", "")
                faults.append(("warning", f"agent:{name}",
                               f"Agent {name} in {status.upper()} state — task: {task}"))
                continue
            # Stale heartbeat detection — skip user-initiated agents.
            if name in _USER_INITIATED_AGENTS:
                continue
            interval = float(entry.get("interval_sec") or 0)
            last_hb  = float(entry.get("last_heartbeat_ts") or 0)
            if interval > 0 and last_hb > 0:
                age = now - last_hb
                if age > max(180.0, interval * 4):
                    faults.append(("warning", f"agent:{name}:stale",
                                   f"Agent {name} heartbeat STALE — last seen {int(age)}s ago "
                                   f"(interval {int(interval)}s)"))
    except Exception as exc:
        faults.append(("warning", "agent:probe", f"Agent probe error: {exc}"))
    return faults


def _probe_scheduler() -> tuple[str, str, str] | None:
    try:
        rep_path = PROJECT_ROOT / "data" / "training_status_report.json"
        if not rep_path.exists():
            return None
        rep = json.loads(rep_path.read_text(encoding="utf-8") or "{}")
        status = str(rep.get("status", "")).lower()
        if status in ("failed", "error", "crashed"):
            return ("warning", "scheduler:last_run",
                    f"Last scheduled run FAILED — {rep.get('host','?')} · "
                    f"epochs {rep.get('epochs_completed', 0)}/{rep.get('epochs_target','?')}")
        return None
    except Exception as exc:
        return ("warning", "scheduler:probe", f"Scheduler probe error: {exc}")


def _probe_cluster() -> list[tuple[str, str, str]]:
    """Optional: poll the orchestrator for crashed workers. We import lazily
    so this module can be imported in environments that don't have the
    orchestrator wired up."""
    faults: list[tuple[str, str, str]] = []
    try:
        from src.training.distributed.orchestrator import get_orchestrator  # type: ignore
        orch = get_orchestrator()
        if orch is None:
            return faults
        st = orch.get_status() or {}
        for w in (st.get("workers") or []):
            wstate = str(w.get("state", "")).lower()
            if wstate in ("error", "crashed", "failed"):
                wid = w.get("worker_id") or w.get("name") or "?"
                faults.append(("warning", f"cluster:{wid}",
                               f"Cluster worker {wid} CRASHED — {w.get('last_error','')}"))
    except Exception:
        # Orchestrator not available / not wired — silent.
        pass
    return faults


# Roles whose exit is normal (one-shot scripts). They're recorded by the
# supervisor with their own filtering, but we double-belt here in case
# legacy entries are still in process_deaths.json.
_TRANSIENT_DEATH_ROLES = {"training"}


def _probe_recent_deaths() -> tuple[str, str, str] | None:
    """Surface fresh process deaths recorded by debug_supervisor.py.
    A death within the last 10 minutes becomes a CRITICAL banner entry
    so the user sees the role + exit clue immediately, not 30 seconds
    after the next dashboard reload. Transient roles (training, etc.)
    are filtered out — their exits are expected behavior.
    """
    try:
        path = PROJECT_ROOT / "data" / "process_deaths.json"
        if not path.exists():
            return None
        deaths = json.loads(path.read_text(encoding="utf-8") or "[]")
        if not deaths:
            return None
        # Walk newest-first, return the first non-transient fresh death
        for latest in deaths:
            role = latest.get("role", "?")
            if role in _TRANSIENT_DEATH_ROLES:
                continue
            died_at = latest.get("died_at", "")
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(died_at.replace("Z", "+00:00"))
                age_s = (_dt.now(tz=dt.tzinfo) - dt).total_seconds()
            except Exception:
                continue
            if age_s > 600:                # only fresh deaths (< 10 min)
                continue
            clue = (latest.get("exit_clue") or latest.get("last_log_line")
                    or "(no log line)")
            return ("critical", f"death:{role}",
                    f"{role} died {int(age_s)}s ago — {clue[:160]}")
        return None
    except Exception as exc:
        logger.debug("[error_monitor] _probe_recent_deaths: %s", exc)
        return None


_ALL_PROBES = [
    ("parquet_store", _probe_parquet_store),
    ("recent_deaths", _probe_recent_deaths),
    ("duckdb",    _probe_duckdb),
    ("parquet",   _probe_parquet),
    ("fastapi",   _probe_fastapi),
    ("realtime",  _probe_realtime),
    ("scheduler", _probe_scheduler),
]
_ALL_MULTI_PROBES = [
    ("processes", _probe_processes),
    ("agents",    _probe_agents),
    ("cluster",   _probe_cluster),
]


def scan_status_surfaces() -> dict[str, dict[str, Any]]:
    """Probe every status surface. Surface entries auto-heal as soon as
    a probe returns None (we drop them from _state in the same pass).
    Returns a snapshot of just the surface entries (for tests)."""
    now = time.time()
    bad: list[tuple[str, str, str]] = []  # (kind, sig_suffix, sample)

    for _name, fn in _ALL_PROBES:
        try:
            r = fn()
        except Exception as exc:
            logger.debug("[error_monitor] probe %s failed: %s", _name, exc)
            continue
        if r is not None:
            bad.append(r)
    for _name, fn in _ALL_MULTI_PROBES:
        try:
            for r in fn() or []:
                bad.append(r)
        except Exception as exc:
            logger.debug("[error_monitor] multi-probe %s failed: %s", _name, exc)

    surface_file = "monitor/health"
    bad_keys: set[str] = set()
    surface_snapshot: dict[str, dict[str, Any]] = {}

    with _state_lock:
        for kind, sig_suffix, sample in bad:
            sig = _SURFACE_PREFIX + sig_suffix
            # Wipe a stale entry that may exist under a different severity
            # (e.g. WARNING → CRITICAL upgrade) before writing the new one.
            for other_kind in ("critical", "warning"):
                if other_kind != kind:
                    other_key = f"{other_kind}::{surface_file}::{sig}"
                    _state.pop(other_key, None)

            key = f"{kind}::{surface_file}::{sig}"
            # Same CLEAR ALL cool-off semantics as scan() — surface probes
            # would otherwise re-add the just-dismissed entry on the very
            # next poll because the underlying status surface is still
            # reporting the same fault.
            if _is_suppressed(key, now):
                continue
            bad_keys.add(key)
            e = _state.get(key)
            if e is None:
                if len(_state) >= MAX_ENTRIES:
                    oldest = min(_state.items(),
                                 key=lambda kv: kv[1].get("first_seen", 0))
                    _state.pop(oldest[0], None)
                e = {
                    "kind":       kind,
                    "file":       surface_file,
                    "signature":  sig,
                    "sample":     sample[:300],
                    "source":     "surface",
                    "first_seen": now,
                    "last_seen":  now,
                    "count":      1,
                }
                _state[key] = e
            else:
                e["last_seen"] = now
                e["count"]     = int(e.get("count", 0)) + 1
                e["sample"]    = sample[:300]
            surface_snapshot[key] = dict(e)

        # Auto-heal: drop surface entries we no longer flag as bad.
        # A surface entry is identified by its signature carrying _SURFACE_PREFIX.
        stale = [
            k for k, e in _state.items()
            if e.get("source") == "surface" and k not in bad_keys
        ]
        for k in stale:
            _state.pop(k, None)

    _save_state()
    return surface_snapshot


def get_active(severity: str | None = None,
               window_s: float = ACTIVE_WINDOW_S) -> list[dict[str, Any]]:
    """Return the active entries (last_seen within window_s). Sorted by
    severity (critical first), then most-recent."""
    now = time.time()
    with _state_lock:
        rows = []
        for key, e in _state.items():
            if (now - e.get("last_seen", 0)) > window_s:
                continue
            if severity and e.get("kind") != severity:
                continue
            row = dict(e)
            row["key"]      = key
            row["age_s"]    = round(now - e.get("last_seen", 0), 1)
            row["span_s"]   = round(e.get("last_seen", 0) - e.get("first_seen", 0), 1)
            rows.append(row)
    rows.sort(key=lambda r: (
        0 if r.get("kind") == "critical" else 1,
        -r.get("last_seen", 0),
    ))
    return rows


def dismiss(key: str) -> bool:
    """Manual dismiss from the UI — drops one entry by key and suppresses
    re-creation for DISMISS_SUPPRESS_S so the next scan doesn't resurrect
    the entry from the still-fresh underlying log line / status probe."""
    now = time.time()
    with _state_lock:
        existed = key in _state
        _state.pop(key, None)
        _dismissed_until[key] = now + DISMISS_SUPPRESS_S
        _save_state()
    return existed


def dismiss_all() -> int:
    """Drop every entry AND start a cool-off so the next scan doesn't
    immediately resurrect them. Returns count cleared. The cool-off lasts
    DISMISS_SUPPRESS_S; after that, fresh occurrences re-flag normally."""
    now = time.time()
    with _state_lock:
        n = len(_state)
        deadline = now + DISMISS_SUPPRESS_S
        for k in list(_state.keys()):
            _dismissed_until[k] = deadline
        _state.clear()
        _save_state()
    return n


def start_monitor_thread() -> threading.Thread:
    """Idempotent. Background scan every REFRESH_S seconds."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        _load_state()
        def _loop():
            time.sleep(5)  # let the dashboard finish booting
            while True:
                try:
                    scan()
                except Exception as exc:
                    logger.debug("[error_monitor] scan error: %s", exc)
                try:
                    scan_status_surfaces()
                except Exception as exc:
                    logger.debug("[error_monitor] surface scan error: %s", exc)
                time.sleep(REFRESH_S)
        t = threading.Thread(target=_loop, daemon=True, name="error-monitor")
        t.start()
        _thread = t
        return t


# Allow CLI inspection: `python -m src.dashboard.error_monitor`
if __name__ == "__main__":
    _load_state()
    snap = scan()
    rows = get_active()
    print(f"{len(rows)} active entries (within {ACTIVE_WINDOW_S//60} min):")
    for r in rows[:20]:
        print(f"  [{r['kind']:<8}] {r['file']:<24} ×{r['count']:<4}"
              f" age {r['age_s']:>6.1f}s  {r['sample'][:120]}")
