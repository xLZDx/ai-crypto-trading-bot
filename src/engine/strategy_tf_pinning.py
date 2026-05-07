"""
strategy_tf_pinning — auto + manual timeframe assignments per strategy.

Phase A of the institutional roadmap. PR 4's Stability heatmap already
identifies the most-stable TF per strategy from walk-forward backtests;
this module persists those assignments and exposes them to:
  - the live bot (so each strategy's signal generation routes to its
    most-stable TF's data + per-TF model)
  - the dashboard's Strategies card (so operators can see + override)

Resolution order (highest priority first):
  1. operator override (manually pinned via dashboard) — written to
     data/strategy_tf_pinning.json under "manual"
  2. auto pin (from the latest multi-TF backtest's best_tf) — written by
     the pipeline orchestrator post-backtest under "auto"
  3. registry default (currently always 1h — the historical canonical TF)

File shape (data/strategy_tf_pinning.json):
    {
      "auto":   {"RSI_MeanReversion": "4h", "MACD_Momentum": "1h", ...},
      "manual": {"VWAP_Reversion":    "1d"},
      "updated_at": "2026-05-07T12:00:00+00:00"
    }
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNING_PATH = PROJECT_ROOT / "data" / "strategy_tf_pinning.json"

DEFAULT_TF = "1h"


def _empty_state() -> dict:
    return {"auto": {}, "manual": {}, "updated_at": ""}


def read_state() -> dict:
    """Return the pinning state. Always returns a dict with auto+manual
    keys, even when the file is missing or malformed."""
    state = read_json(str(PINNING_PATH), default=_empty_state()) or _empty_state()
    if not isinstance(state, dict):
        state = _empty_state()
    state.setdefault("auto", {})
    state.setdefault("manual", {})
    return state


def write_state(state: dict) -> None:
    state.setdefault("auto", {})
    state.setdefault("manual", {})
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(str(PINNING_PATH), state)


def get_pinned_tf(strategy: str, default: str = DEFAULT_TF) -> str:
    """Return the active TF for a strategy.

    Resolution: manual override > auto pin > default. Used by main.py /
    strategy_registry to decide which per-TF model + features feed each
    strategy's signal generation.
    """
    state = read_state()
    manual = state.get("manual") or {}
    auto = state.get("auto") or {}
    return manual.get(strategy) or auto.get(strategy) or default


def get_all_pins() -> dict[str, dict[str, str]]:
    """Return {strategy: {auto, manual, effective}} for every strategy
    that has any pin (auto or manual). The dashboard reads this to
    render the TF column on the Strategies card."""
    state = read_state()
    auto = state.get("auto") or {}
    manual = state.get("manual") or {}
    out: dict[str, dict[str, str]] = {}
    for s in set(auto.keys()) | set(manual.keys()):
        out[s] = {
            "auto":      auto.get(s, ""),
            "manual":    manual.get(s, ""),
            "effective": manual.get(s) or auto.get(s) or DEFAULT_TF,
        }
    return out


def set_manual_pin(strategy: str, tf: str | None) -> dict:
    """Set or clear the manual override for one strategy. Pass tf=None
    (or empty string) to clear and fall back to auto / default."""
    state = read_state()
    manual = dict(state.get("manual") or {})
    if not tf:
        manual.pop(strategy, None)
    else:
        manual[strategy] = str(tf)
    state["manual"] = manual
    write_state(state)
    return state


def update_auto_pins(best_tf_by_strategy: dict[str, str]) -> dict:
    """Replace the auto-pin map with the latest backtest's best_tf data.
    Called by the pipeline orchestrator post-backtest. Clears any auto
    pins for strategies the new backtest didn't cover (safer than
    keeping stale assignments around)."""
    state = read_state()
    state["auto"] = {k: str(v) for k, v in (best_tf_by_strategy or {}).items() if v}
    write_state(state)
    logger.info("[tf_pinning] auto-pins updated for %d strategies",
                len(state["auto"]))
    return state


__all__ = [
    "PINNING_PATH", "DEFAULT_TF",
    "read_state", "write_state",
    "get_pinned_tf", "get_all_pins",
    "set_manual_pin", "update_auto_pins",
]
