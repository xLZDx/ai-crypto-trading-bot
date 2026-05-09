"""File-based pub-sub topics — Kafka-inspired but lightweight.

Each topic is a directory containing:
  - log-YYYYMMDD.jsonl              (append-only, one JSON object per line)
  - consumers/<consumer-id>.offset  (last-committed offset per consumer)

Producers append a line to today's log file. Consumers tail by offset and
commit progress when done. No broker process — the topic IS the directory.

Why we built this instead of a real broker
------------------------------------------
The trading bot already runs entirely on the local D: drive with a strict
disk-over-RAM preference. A real broker (Kafka/RabbitMQ/Redis Streams)
would add a daemon to monitor, network ports to manage, and a learning
curve. For our scale (≤ thousands of events/min across a handful of
producers and consumers, all on one or two laptops), a directory of
JSONL files with daily rotation gives us:

  - durability               (atomic writes via filelock from safe_json)
  - replay                   (consumer just re-reads from offset 0)
  - inspection               (any text editor reads the log)
  - zero ops                 (no daemon to keep up)
  - process isolation        (producers and consumers don't share memory)

Trade-offs vs Kafka
-------------------
  - polling, not push  → STAR latency: ~POLL_INTERVAL_S (default 5 s)
  - no replication     → if the disk dies, the topic is gone (we accept this)
  - no consumer groups → if you want fan-out, give consumers different IDs
  - no compaction      → log files grow forever (retention is layer-5 problem)

Concurrency
-----------
Multiple producers can append to the same log file concurrently — we hold
the file's filelock during append so the JSONL line ordering is consistent.
Multiple consumers reading the same topic don't interfere; each has its
own offset file.

Offsets are byte positions in the daily log file. To survive day rollover,
the offset stores BOTH the date string and the byte position:
  {"date": "20260510", "byte_offset": 1234}

When a consumer's stored date is older than today, it advances through
each intermediate day's log in order before reading today's.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)

# Project root (4 levels up from this file: src/orchestration/topics.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPICS_DIR   = PROJECT_ROOT / "data" / "topics"

# Canonical topic names — referenced by supervisors and dashboard.
# (Pre-declared so typos in topic names raise on import, not silently
# create a new orphan directory.)
TOPIC_TRAINING_REQUESTS    = "training.requests"
TOPIC_TRAINING_EVENTS      = "training.events"
TOPIC_TRAINING_CHECKPOINTS = "training.checkpoints"
TOPIC_BACKTEST_REQUESTS    = "backtest.requests"
TOPIC_BACKTEST_EVENTS      = "backtest.events"
TOPIC_SERVICE_HEARTBEATS   = "service.heartbeats"
TOPIC_SERVICE_ALERTS       = "service.alerts"

KNOWN_TOPICS = (
    TOPIC_TRAINING_REQUESTS, TOPIC_TRAINING_EVENTS, TOPIC_TRAINING_CHECKPOINTS,
    TOPIC_BACKTEST_REQUESTS, TOPIC_BACKTEST_EVENTS,
    TOPIC_SERVICE_HEARTBEATS, TOPIC_SERVICE_ALERTS,
)

_DATE_FMT     = "%Y%m%d"
_LOG_GLOB_RE  = re.compile(r"^log-(\d{8})\.jsonl$")


@dataclass(frozen=True)
class TopicEntry:
    """One record yielded by Topic.tail()."""
    date: str           # YYYYMMDD
    byte_offset: int    # byte position AT END of this entry's line (commit-this)
    payload: dict       # parsed JSON


@dataclass(frozen=True)
class TopicStats:
    """Snapshot for the dashboard Topics card."""
    name: str
    bytes_total: int
    lines_total: int
    days_present: int
    last_append_ts: Optional[float]    # epoch seconds; None if empty


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime(_DATE_FMT)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, _DATE_FMT).replace(tzinfo=timezone.utc)


class Topic:
    """One topic = one directory under data/topics/.

    Thread-safe for concurrent producers (filelock guards each append).
    Multiple consumers with distinct IDs can tail simultaneously without
    blocking each other or the producers.
    """

    def __init__(self, name: str):
        if name not in KNOWN_TOPICS:
            # Soft warning, not hard fail — useful for dynamic topics in
            # tests. Production code uses the constants above.
            logger.warning("Topic %r is not in KNOWN_TOPICS; create-on-write OK for tests", name)
        self.name = name
        self.dir  = TOPICS_DIR / name
        self.consumers_dir = self.dir / "consumers"
        # Don't materialise on construct — append() does it lazily.
        self._append_lock = threading.Lock()

    # ── helpers ─────────────────────────────────────────────────────────

    def _log_path(self, date_str: str) -> Path:
        return self.dir / f"log-{date_str}.jsonl"

    def _ensure_dirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.consumers_dir.mkdir(parents=True, exist_ok=True)

    def _all_dates(self) -> list[str]:
        """Sorted list of YYYYMMDD strings present on disk."""
        if not self.dir.exists():
            return []
        out = []
        for p in self.dir.iterdir():
            m = _LOG_GLOB_RE.match(p.name)
            if m:
                out.append(m.group(1))
        out.sort()
        return out

    def _consumer_offset_path(self, consumer: str) -> Path:
        return self.consumers_dir / f"{consumer}.offset"

    # ── producer API ────────────────────────────────────────────────────

    def append(self, payload: dict) -> int:
        """Append one JSON-serialisable record. Returns the byte offset
        within today's log file pointing AT THE END of this record's line
        (so a subsequent consumer commit at this value won't re-read it).

        Atomic via filelock — concurrent producers don't interleave lines.
        Adds a `_ts` field if the caller didn't supply one.
        """
        self._ensure_dirs()
        if "_ts" not in payload:
            payload = {**payload, "_ts": time.time()}
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        path = self._log_path(_today_str())
        lock = FileLock(str(path) + ".lock", timeout=5)
        with self._append_lock, lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                end_offset = f.tell()
        return end_offset

    # ── consumer API ────────────────────────────────────────────────────

    def get_offset(self, consumer: str) -> dict:
        """Return the saved offset for `consumer`. Default = beginning
        of the earliest day on disk, or today if topic is empty.
        Format: {"date": "YYYYMMDD", "byte_offset": N}"""
        path = self._consumer_offset_path(consumer)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        dates = self._all_dates()
        return {"date": dates[0] if dates else _today_str(), "byte_offset": 0}

    def commit(self, consumer: str, date: str, byte_offset: int) -> None:
        """Persist the offset after processing. Atomic write."""
        self._ensure_dirs()
        path = self._consumer_offset_path(consumer)
        payload = {"date": date, "byte_offset": int(byte_offset),
                   "_committed_ts": time.time()}
        lock = FileLock(str(path) + ".lock", timeout=5)
        with lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)

    def tail(self, consumer: str, batch: int = 100) -> Iterator[TopicEntry]:
        """Yield up to `batch` new TopicEntry records since the consumer's
        last commit. The caller is expected to commit() after each entry
        is processed (or after the whole batch — at-least-once).

        Walks the daily log files in date order. When the consumer's
        stored date is older than today, advances day-by-day until caught
        up. Within a day, reads from byte_offset to EOF.
        """
        offset = self.get_offset(consumer)
        cur_date = offset["date"]
        cur_pos  = int(offset["byte_offset"])
        emitted = 0
        all_dates = self._all_dates()
        if not all_dates:
            return
        # Skip dates strictly before cur_date — consumer already past them.
        try:
            start_idx = next(i for i, d in enumerate(all_dates) if d >= cur_date)
        except StopIteration:
            return  # all logs older than consumer's offset → nothing to emit
        for d in all_dates[start_idx:]:
            log_path = self._log_path(d)
            try:
                f = open(log_path, "rb")
            except FileNotFoundError:
                continue
            try:
                if d == cur_date:
                    f.seek(cur_pos)
                while emitted < batch:
                    line = f.readline()
                    if not line:
                        break
                    end = f.tell()
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        logger.warning("topic %s log %s line@%d malformed: %s",
                                       self.name, d, end, exc)
                        continue
                    yield TopicEntry(date=d, byte_offset=end, payload=payload)
                    emitted += 1
            finally:
                f.close()
            if emitted >= batch:
                return

    # ── inspection (for dashboard Topics card) ──────────────────────────

    def stats(self) -> TopicStats:
        """Bytes + line count + day count + last append time. Reads the
        directory fresh each call (cheap; we only have a handful of files
        per topic given daily rotation)."""
        if not self.dir.exists():
            return TopicStats(name=self.name, bytes_total=0, lines_total=0,
                              days_present=0, last_append_ts=None)
        bytes_total = 0
        lines_total = 0
        last_mtime: Optional[float] = None
        days = 0
        for d in self._all_dates():
            p = self._log_path(d)
            try:
                st = p.stat()
                bytes_total += st.st_size
                if last_mtime is None or st.st_mtime > last_mtime:
                    last_mtime = st.st_mtime
                # Cheap line count via newline count.
                with open(p, "rb") as f:
                    lines_total += sum(1 for _ in f)
                days += 1
            except OSError:
                continue
        return TopicStats(name=self.name, bytes_total=bytes_total,
                          lines_total=lines_total, days_present=days,
                          last_append_ts=last_mtime)


# ── module-level convenience ───────────────────────────────────────────

_topic_cache: dict[str, Topic] = {}
_topic_cache_lock = threading.Lock()


def topic(name: str) -> Topic:
    """Get-or-create a Topic instance. Cached so callers can share."""
    with _topic_cache_lock:
        t = _topic_cache.get(name)
        if t is None:
            t = Topic(name)
            _topic_cache[name] = t
        return t


def all_topic_stats() -> list[TopicStats]:
    """One-shot snapshot of every known topic — for the dashboard card.
    Always returns one entry per KNOWN_TOPIC even if the directory hasn't
    materialised yet (so the card shows zeros instead of dropping rows).
    """
    return [topic(name).stats() for name in KNOWN_TOPICS]
