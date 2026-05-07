"""
live_news_buffer — in-memory rolling cache of recent news for live inference.

Phase D of the institutional roadmap. Today add_news_sentiment() queries
the news Parquet partitions on every signal cycle (~100-500ms cold-start
DuckDB cost per call). The bot evaluates ~20 symbols × N strategies per
minute, so that overhead becomes the dominant feature-build cost.

This module:
  - Loads the last `window_hours` of news rows ONCE into a Pandas DF on
    background-thread startup.
  - Refreshes that snapshot every `refresh_seconds` from the same
    Parquet partitions (which the GDELT / Reddit / CryptoCompare
    scrapers update continuously).
  - Exposes a thread-safe `get_snapshot()` that returns the cached DF in
    O(1) — no I/O on the hot path.

add_news_sentiment() picks up the buffer when present; otherwise falls
back to its existing per-call DuckDB query (training / backtest path).

Public surface:
  start_buffer(window_hours=48, refresh_seconds=300) -> LiveNewsBuffer
  get_active_buffer() -> LiveNewsBuffer | None
  stop_buffer()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton — only one buffer per process.
_active_buffer: Optional["LiveNewsBuffer"] = None
_active_lock = threading.Lock()


class LiveNewsBuffer:
    def __init__(self, window_hours: int = 48, refresh_seconds: int = 300):
        self.window_hours = int(window_hours)
        self.refresh_seconds = int(refresh_seconds)
        self._snapshot = None  # type: ignore  # pd.DataFrame | None
        self._snapshot_ts: float = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._refresh_count = 0
        self._last_error: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "LiveNewsBuffer":
        """Kick off the background refresher. Idempotent — calling start
        twice is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._refresh_once_safe()   # populate before returning so the
                                    # first hot-path call sees real data
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="live-news-buffer")
        self._thread.start()
        logger.info("[live_news_buffer] started (window=%dh, refresh=%ds)",
                    self.window_hours, self.refresh_seconds)
        return self

    def stop(self) -> None:
        self._stop.set()
        # Don't join — daemon thread will die with process. Joining could
        # block the bot's shutdown if a refresh is mid-flight.

    def get_snapshot(self):
        """Return the cached DataFrame snapshot. Thread-safe; never blocks
        on I/O. Returns None if the first refresh hasn't completed yet
        (rare — start() does an inline refresh before returning)."""
        with self._lock:
            return self._snapshot

    def status(self) -> dict:
        with self._lock:
            return {
                "ready":             self._snapshot is not None,
                "rows":              0 if self._snapshot is None else int(len(self._snapshot)),
                "snapshot_age_s":    round(time.time() - self._snapshot_ts, 1)
                                      if self._snapshot_ts else None,
                "refresh_count":     self._refresh_count,
                "window_hours":      self.window_hours,
                "refresh_seconds":   self.refresh_seconds,
                "last_error":        self._last_error,
            }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _refresh_once_safe(self) -> None:
        try:
            self._refresh_once()
        except Exception as exc:
            logger.warning("[live_news_buffer] refresh failed: %s", exc)
            self._last_error = f"{type(exc).__name__}: {exc}"

    def _refresh_once(self) -> None:
        # Late import — avoids cycles and lets the buffer module load even
        # if duckdb / pyarrow aren't installed (we just won't refresh).
        from src.analysis.feature_reader import load_news_recent
        rows = load_news_recent(hours=self.window_hours)
        if rows is None:
            return
        try:
            import pandas as pd
            df = pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as exc:
            self._last_error = f"pd.DataFrame: {exc}"
            return
        with self._lock:
            self._snapshot = df
            self._snapshot_ts = time.time()
            self._refresh_count += 1
            self._last_error = ""

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Wait first, refresh second — start() did the initial refresh.
            self._stop.wait(timeout=self.refresh_seconds)
            if self._stop.is_set():
                break
            self._refresh_once_safe()


# ── Module-level helpers ──────────────────────────────────────────────────────

def start_buffer(window_hours: int = 48,
                 refresh_seconds: int = 300) -> LiveNewsBuffer:
    """Start (or return existing) module-level singleton buffer. The bot's
    main loop calls this once at startup."""
    global _active_buffer
    with _active_lock:
        if _active_buffer is None:
            _active_buffer = LiveNewsBuffer(window_hours=window_hours,
                                             refresh_seconds=refresh_seconds)
            _active_buffer.start()
        return _active_buffer


def get_active_buffer() -> Optional[LiveNewsBuffer]:
    """Return the running buffer, or None if start_buffer() was never
    called. add_news_sentiment() uses this to decide whether to skip the
    DuckDB query path."""
    with _active_lock:
        return _active_buffer


def stop_buffer() -> None:
    global _active_buffer
    with _active_lock:
        if _active_buffer is not None:
            _active_buffer.stop()
            _active_buffer = None


__all__ = [
    "LiveNewsBuffer", "start_buffer", "get_active_buffer", "stop_buffer",
]
