"""Hot-reloadable runtime overrides.

The dashboard's Risk sub-tab writes to data/runtime_overrides.json. The
trading bot reads it via this module on every loop. Cached for 1 second
so the per-tick read is essentially free, but new values still propagate
within ~1s of the user pressing Save in the UI.

Schema:
    max_position_usdt:           float | None  — caps trade_amount after
                                                  Kelly + GARCH + OFT weight
    scalping_disabled_symbols:   list[str]     — skip the 1m scalp path
                                                  for these symbols
    trailing_stop_pct_scalping:  float | None  — overrides
                                                  DEFAULT_TRAILING_STOP_PCT
                                                  for the scalping path
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = PROJECT_ROOT / "data" / "runtime_overrides.json"

_DEFAULTS = {
    "max_position_usdt":          None,
    "scalping_disabled_symbols":  [],
    "trailing_stop_pct_scalping": None,
}

_TTL_S = 1.0
_cache: dict = {"data": dict(_DEFAULTS), "expires": 0.0, "mtime": 0.0}


def _load() -> dict:
    out = dict(_DEFAULTS)
    try:
        if OVERRIDES_PATH.exists():
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            for k in _DEFAULTS:
                if k in raw:
                    out[k] = raw[k]
    except Exception as exc:
        logger.debug("[runtime_overrides] load failed: %s", exc)
    return out


def get() -> dict:
    """Cached read. TTL=1s so the bot loop incurs almost no I/O even at 60 Hz."""
    now = time.monotonic()
    if now < _cache["expires"]:
        return _cache["data"]
    try:
        mtime = OVERRIDES_PATH.stat().st_mtime if OVERRIDES_PATH.exists() else 0.0
    except Exception:
        mtime = 0.0
    if mtime != _cache["mtime"]:
        _cache["data"] = _load()
        _cache["mtime"] = mtime
    _cache["expires"] = now + _TTL_S
    return _cache["data"]


def is_scalping_disabled(symbol: str) -> bool:
    """True iff the user has flagged this symbol on the scalping kill-list.

    Accepts both 'BTC/USDT' and 'BTC_USDT' style — normalises before
    comparison so callers don't have to remember which form is on disk."""
    if not symbol:
        return False
    norm = symbol.replace("_", "/").upper()
    disabled = get().get("scalping_disabled_symbols") or []
    return any(norm == s.replace("_", "/").upper() for s in disabled)


def max_position_cap() -> float | None:
    return get().get("max_position_usdt")


def trailing_stop_pct_scalping(default: float) -> float:
    """Returns the override value, or `default` when no override set."""
    val = get().get("trailing_stop_pct_scalping")
    try:
        return float(val) if val is not None else float(default)
    except (TypeError, ValueError):
        return float(default)
