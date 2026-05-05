"""
DataSourceConnector — the base contract every external feed implements.

A connector promises to:
  • Self-describe (name, priority, requires_auth, host).
  • Tell the orchestrator whether it can run *right now* (is_available).
  • Pull historical data into the local DB (`pull_history`).
  • Optionally maintain a realtime stream (`realtime_loop`); base class
    falls back to a periodic re-pull every `default_poll_interval_sec`.

Storage convention:
  • Hot path:  QuestDB ILP, table `src_{name}` (one table per source).
  • Cold path: Parquet via `ParquetStore.ingest_csv(path, symbol,
    timeframe='news' or similar)` for sources that don't fit the OHLCV
    schema; OHLCV-shaped sources write to the standard `market_data` table.

Rate limiting: never call `requests.get()` directly. Use
    `with rate_limiter.get_limiter(self.host).acquire(weight=1):`
or the `@rate_limited(host)` decorator.
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ConnectorMeta:
    """Static description used by the orchestrator + dashboard."""
    name:           str
    host:           str             # for rate-limiter lookup
    priority:       int = 1         # 0 = core, 1 = auxiliary, 2 = premium
    requires_auth:  bool = False
    default_poll_interval_sec: int = 3600
    description:    str = ""
    category:       str = "market"  # market | news | onchain | macro | derivatives | sentiment


class DataSourceConnector(abc.ABC):
    """Base class. Subclasses define a class attribute `META`."""

    META: ConnectorMeta = ConnectorMeta(name="base", host="example.com")

    def __init__(self, **opts):
        self.opts = opts
        self._stop = False

    @property
    def name(self) -> str:
        return self.META.name

    # ── Connector hooks ─────────────────────────────────────────────────

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the source is usable right now (auth, network, etc.)."""

    @abc.abstractmethod
    def pull_history(self, *, since=None, until=None, **kw) -> int:
        """Bulk-load historical data into the DB. Returns rows written."""

    def realtime_loop(self, *, poll_interval_sec: int | None = None,
                      on_stop: Callable[[], bool] | None = None) -> None:
        """Default realtime: periodic re-pull. Override for true streaming."""
        interval = poll_interval_sec or self.META.default_poll_interval_sec
        logger.info("[%s] realtime poll loop, interval=%ds", self.name, interval)
        while not self._stop:
            if on_stop and on_stop():
                return
            try:
                n = self.pull_history()
                logger.debug("[%s] poll wrote %d rows", self.name, n)
            except Exception as exc:
                logger.warning("[%s] poll error: %s", self.name, exc)
            for _ in range(interval):
                if self._stop or (on_stop and on_stop()):
                    return
                time.sleep(1)

    def stop(self) -> None:
        self._stop = True

    # ── Helpers for subclasses ─────────────────────────────────────────

    def _http_get(self, url: str, *, weight: int = 1, **kwargs):
        """Rate-limited GET. Returns the requests.Response or None on error."""
        import requests
        from src.data_ingestion.rate_limiter import get_limiter
        limiter = get_limiter(self.META.host)
        with limiter.acquire(weight=weight):
            try:
                r = requests.get(url, timeout=kwargs.pop("timeout", 15), **kwargs)
                limiter.react_to_response(r)
                return r
            except requests.RequestException as exc:
                logger.warning("[%s] GET %s failed: %s", self.name, url, exc)
                return None

    def _qdb(self):
        from src.database.parquet_client import get_client
        return get_client()

    def _parquet(self):
        from src.database.parquet_store import get_store
        return get_store()


__all__ = ["DataSourceConnector", "ConnectorMeta"]
