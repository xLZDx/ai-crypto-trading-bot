"""
Dual-balance state — Phase 9.

Two completely separate state objects:

    REAL   — read from Binance Spot/Futures balance API on each refresh
             and persisted to data/balance_real.json
    VIRTUAL — managed by the simulator + RL training; persisted to
              data/balance_virtual.json

Both use safe_json.read_json/write_json (file lock + atomic writes) so
multiple processes (bot, dashboard, training, simulator) never see a
half-written file.

The dashboard's REAL vs TEST/TRAIN tab switcher reads from these two
files and shows whichever is selected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from src.utils.safe_json import read_json, write_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_PATH    = PROJECT_ROOT / "data" / "balance_real.json"
VIRTUAL_PATH = PROJECT_ROOT / "data" / "balance_virtual.json"


@dataclass
class BalanceSnapshot:
    """Common shape used by both real and virtual."""
    mode:        str = "real"        # "real" | "virtual"
    timestamp:   str = ""
    cash_usdt:   float = 0.0
    holdings:    dict = field(default_factory=dict)   # symbol -> qty
    equity_usdt: float = 0.0
    pnl_24h:     float = 0.0
    drawdown_pct: float = 0.0
    trade_count_24h: int = 0


def _empty(mode: str) -> dict:
    return BalanceSnapshot(mode=mode, timestamp=datetime.now(timezone.utc).isoformat()).__dict__


# ─── Real (live Binance) ─────────────────────────────────────────────────

def write_real(snapshot: dict) -> None:
    snapshot = {**_empty("real"), **(snapshot or {})}
    snapshot["mode"] = "real"
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(str(REAL_PATH), snapshot)


def read_real() -> dict:
    return read_json(str(REAL_PATH), default=_empty("real")) or _empty("real")


def refresh_real_from_binance(order_manager=None) -> dict:
    """Pull live Spot+Futures balances and save to disk.

    Pass an existing `OrderManager` instance, or one will be created.
    Falls back to the last cached snapshot if Binance is unreachable.
    """
    try:
        if order_manager is None:
            from src.engine.order_manager import OrderManager
            order_manager = OrderManager()

        usdt = float(order_manager.get_balance("USDT") or 0)
        holdings = {}
        for asset in ("BTC", "ETH", "SOL", "ADA"):
            try:
                qty = float(order_manager.get_balance(asset) or 0)
                if qty > 0:
                    holdings[asset] = qty
            except Exception:
                pass

        snapshot = {
            "mode": "real",
            "cash_usdt": usdt,
            "holdings":  holdings,
            "equity_usdt": usdt,   # ignoring holdings mark-to-market here
        }
        write_real(snapshot)
        return snapshot
    except Exception as exc:
        logger.warning("[balance_real] refresh failed: %s — keeping cache.", exc)
        return read_real()


# ─── Virtual (simulator / RL training) ───────────────────────────────────

def write_virtual(snapshot: dict) -> None:
    snapshot = {**_empty("virtual"), **(snapshot or {})}
    snapshot["mode"] = "virtual"
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(str(VIRTUAL_PATH), snapshot)


def read_virtual() -> dict:
    return read_json(str(VIRTUAL_PATH), default=_empty("virtual")) or _empty("virtual")


def reset_virtual(initial_cash: float = 100_000.0) -> dict:
    """Reset the virtual balance to a fresh state with `initial_cash` as
    a single seed deposit. Subsequent operator deposits are tracked in
    the deposits[] array; the balance never auto-syncs from the exchange."""
    now = datetime.now(timezone.utc).isoformat()
    snap = {
        "mode": "virtual",
        "cash_usdt": float(initial_cash),
        "holdings":  {},
        "equity_usdt": float(initial_cash),
        "pnl_24h": 0.0,
        "drawdown_pct": 0.0,
        "trade_count_24h": 0,
        "deposits": [{"ts": now, "amount": float(initial_cash), "note": "seed"}],
        "revenue_total": 0.0,   # cumulative closed-trade pnl_usdt (paper)
    }
    write_virtual(snap)
    return snap


def add_deposit(amount: float, note: str = "") -> dict:
    """Operator manually adds funds to the virtual balance. Updates
    cash_usdt and equity_usdt by `amount` and appends to deposits[]."""
    snap = read_virtual()
    if "deposits" not in snap or not isinstance(snap.get("deposits"), list):
        # Migrate older balance files: treat existing cash as a single
        # implicit deposit so the math stays self-consistent.
        snap["deposits"] = [{
            "ts":     snap.get("timestamp")
                      or datetime.now(timezone.utc).isoformat(),
            "amount": float(snap.get("cash_usdt", 0)),
            "note":   "migrated-from-cash",
        }]
    snap["deposits"].append({
        "ts":     datetime.now(timezone.utc).isoformat(),
        "amount": float(amount),
        "note":   str(note)[:120],
    })
    snap["cash_usdt"]   = float(snap.get("cash_usdt",   0)) + float(amount)
    snap["equity_usdt"] = float(snap.get("equity_usdt", 0)) + float(amount)
    write_virtual(snap)
    return snap


def add_paper_pnl(pnl_usdt: float) -> dict:
    """Apply a closed paper trade's PnL to the virtual balance.
    Updates cash_usdt and revenue_total. Used by the paper booker so the
    virtual balance accumulates only from closed trades + manual deposits."""
    snap = read_virtual()
    snap["cash_usdt"]     = float(snap.get("cash_usdt", 0))     + float(pnl_usdt)
    snap["equity_usdt"]   = float(snap.get("equity_usdt", 0))   + float(pnl_usdt)
    snap["revenue_total"] = float(snap.get("revenue_total", 0)) + float(pnl_usdt)
    write_virtual(snap)
    return snap


def compute_summary() -> dict:
    """Decompose the virtual balance into operator deposits vs trading
    revenue so the dashboard can show P&L cleanly:
        equity         = cash + holdings_value
        deposits_total = sum(deposits[].amount)
        revenue_total  = cumulative pnl from closed paper trades
        pnl            = equity - deposits_total      (== revenue when
                                                      no positions open)

    Migration helper: pre-PR-6 balance files have no deposits[] array.
    For those, we report deposits_total = cash so pnl starts at 0 (which
    is the truthful state — the operator hasn't yet recorded a deposit).
    """
    snap = read_virtual()
    deposits = snap.get("deposits") or []
    revenue_total = float(snap.get("revenue_total", 0) or 0)
    equity = float(snap.get("equity_usdt", 0) or 0)
    cash   = float(snap.get("cash_usdt", 0) or 0)
    if deposits:
        deposits_total = float(sum(d.get("amount", 0) or 0 for d in deposits))
        deposits_count = len(deposits)
    else:
        # Implicit deposit = current cash. Avoids showing pnl = $cash on
        # legacy balance files. First explicit add_deposit() will replace
        # this with the real seed entry.
        deposits_total = cash
        deposits_count = 0
    return {
        "mode":           snap.get("mode", "virtual"),
        "cash":           cash,
        "equity":         equity,
        "deposits_total": deposits_total,
        "deposits_count": deposits_count,
        "revenue_total":  revenue_total,
        "pnl":            round(equity - deposits_total, 6),
    }


__all__ = [
    "BalanceSnapshot",
    "read_real", "write_real", "refresh_real_from_binance",
    "read_virtual", "write_virtual", "reset_virtual",
    "add_deposit", "add_paper_pnl", "compute_summary",
    "REAL_PATH", "VIRTUAL_PATH",
]
