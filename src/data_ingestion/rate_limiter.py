"""
Rate limiter for Binance public REST endpoints.

Binance Spot REST limits (per IP) — see
https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-info

  - 1,200 *requests* / minute   (X-MBX-USED-WEIGHT-1M counter, weight 1 each)
  - 6,000 *weight* / minute     (some endpoints cost more, e.g. /klines limit=1000 = 5)
  - 50    orders / second        (irrelevant for read-only data ingest)

We use a simple token-bucket per (host, window) tuple, plus reactive backoff
when Binance returns:
  - 429  "Too Many Requests"  → sleep `Retry-After` seconds
  - 418  "I'm a teapot"       → IP-banned. Sleep retry_after, then keep going.

This module is **thread-safe** so the parallelized archive downloader and
the realtime gap-fill REST top-up share the same budget.

Usage:
    limiter = get_limiter("binance.com")
    with limiter.acquire(weight=1):
        r = requests.get(url, timeout=10)

Or as a decorator:
    @rate_limited("binance.com", weight=5)
    def fetch_klines(...): ...
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# Conservative defaults: stay well below the published cap so concurrent
# tools (archive downloader, REST top-up, governance orchestrator) can share.
DEFAULT_WEIGHT_PER_MIN = 5_000   # 80% of 6,000 cap
DEFAULT_REQ_PER_MIN    = 1_000   # 80% of 1,200 cap

_DEFAULT_HOSTS = {
    "binance.com":          {"weight_per_min": DEFAULT_WEIGHT_PER_MIN,
                             "req_per_min":    DEFAULT_REQ_PER_MIN},
    "data.binance.vision":  {"weight_per_min": 99_999, "req_per_min": 600},  # CDN, generous
    "fapi.binance.com":     {"weight_per_min": DEFAULT_WEIGHT_PER_MIN,
                             "req_per_min":    DEFAULT_REQ_PER_MIN},
    "api.coingecko.com":    {"weight_per_min": 99_999, "req_per_min": 30},   # 30/min free
    "api.bybit.com":        {"weight_per_min": 99_999, "req_per_min": 600},
    "www.okx.com":          {"weight_per_min": 99_999, "req_per_min": 600},
    "api.exchange.coinbase.com": {"weight_per_min": 99_999, "req_per_min": 600},
    "api.kraken.com":       {"weight_per_min": 99_999, "req_per_min": 60},
    "api.alternative.me":   {"weight_per_min": 99_999, "req_per_min": 60},
    "api.stlouisfed.org":   {"weight_per_min": 99_999, "req_per_min": 120},
    "api.llama.fi":         {"weight_per_min": 99_999, "req_per_min": 60},
    "min-api.cryptocompare.com": {"weight_per_min": 99_999, "req_per_min": 100},
    "open-api.coinglass.com": {"weight_per_min": 99_999, "req_per_min": 30},
}


class RateLimiter:
    """Sliding-window token bucket. One instance per host.

    Thread-safe: multiple worker threads can call `acquire()` concurrently.
    """

    def __init__(self, host: str, weight_per_min: int, req_per_min: int):
        self.host = host
        self.weight_per_min = int(weight_per_min)
        self.req_per_min    = int(req_per_min)
        self._lock = threading.Lock()
        # deque of (epoch_ts, weight) for events in the last 60 s
        self._events: deque[tuple[float, int]] = deque()
        self._banned_until: float = 0.0

    def _evict(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _budget_used(self, now: float) -> tuple[int, int]:
        self._evict(now)
        weight_used = sum(w for _, w in self._events)
        req_used    = len(self._events)
        return weight_used, req_used

    @contextmanager
    def acquire(self, weight: int = 1):
        """Block until budget is available; record the call on exit."""
        while True:
            now = time.time()
            if now < self._banned_until:
                sleep_for = self._banned_until - now
                logger.warning("[ratelimit %s] under ban -- sleeping %.1fs", self.host, sleep_for)
                time.sleep(min(sleep_for, 60))
                continue
            with self._lock:
                w, r = self._budget_used(now)
                if (w + weight) <= self.weight_per_min and (r + 1) <= self.req_per_min:
                    self._events.append((now, weight))
                    break
                # Compute how long to wait so the oldest event ages out.
                if self._events:
                    wait = max(0.05, 60.0 - (now - self._events[0][0]))
                else:
                    wait = 0.05
            time.sleep(min(wait, 5.0))
        try:
            yield
        except Exception:
            raise

    def react_to_response(self, response) -> None:
        """Call after every HTTP response — reads Retry-After / X-MBX-USED-WEIGHT headers."""
        if response is None:
            return
        try:
            code = getattr(response, "status_code", None)
            if code in (429, 418):
                ra = response.headers.get("Retry-After", "30")
                try:
                    secs = float(ra)
                except ValueError:
                    secs = 30.0
                self._banned_until = time.time() + secs
                logger.warning("[ratelimit %s] %d Too Many -- banned %.0fs", self.host, code, secs)
        except Exception:
            pass


# ─── Process-wide registry ─────────────────────────────────────────────────

_REGISTRY: dict[str, RateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_limiter(host: str) -> RateLimiter:
    """Return the shared limiter for `host`. Auto-creates from defaults."""
    with _REGISTRY_LOCK:
        if host not in _REGISTRY:
            cfg = _DEFAULT_HOSTS.get(host, {"weight_per_min": 1000, "req_per_min": 100})
            _REGISTRY[host] = RateLimiter(host, **cfg)
        return _REGISTRY[host]


def rate_limited(host: str, weight: int = 1):
    """Decorator: wraps a function in the host's rate limiter."""
    limiter = get_limiter(host)
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with limiter.acquire(weight=weight):
                resp = fn(*args, **kwargs)
                limiter.react_to_response(resp)
                return resp
        return wrapped
    return deco


def stats() -> dict:
    """Snapshot of current usage — exposed on the dashboard for visibility."""
    out = {}
    now = time.time()
    for host, lim in _REGISTRY.items():
        with lim._lock:
            w, r = lim._budget_used(now)
        out[host] = {
            "weight_used_60s": w,
            "weight_cap_60s":  lim.weight_per_min,
            "req_used_60s":    r,
            "req_cap_60s":     lim.req_per_min,
            "banned_for_sec":  max(0, lim._banned_until - now),
        }
    return out


__all__ = ["RateLimiter", "get_limiter", "rate_limited", "stats"]
