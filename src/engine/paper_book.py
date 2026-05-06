"""
paper_book — book trades against the internal virtual balance instead
of sending them to Binance.

Used when data/control.json's `trade_mode` field is set to "paper". In
that mode the bot still consumes the live Binance feed and generates
signals; orders that would normally hit the exchange land here instead.

Each "paper order" is recorded with the same shape as a real order so
TradeTracker / dashboard layers can read it transparently:
  {
    "id":           "paper-<uuid>",
    "is_paper":     true,
    "symbol":       "BTC/USDT",
    "side":         "BUY",
    "amount":       0.01,
    "price":        81700.0,
    "cost":         817.0,
    "ts":           "2026-05-06T22:30:00+00:00",
    "fee":          {"cost": 0.4, "currency": "USDT"},
    "info":         {"source": "paper_book"},
    "status":       "closed",
  }

The bot's signal-pipeline calls `book_market_order(side, symbol, amt,
price)` to record the entry. When the position is closed (TP/SL/manual)
the bot calls `book_close(symbol, exit_price)` to compute pnl and credit
the virtual balance via dual_balance.add_paper_pnl().
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.safe_json import read_json, write_json
from src.engine.dual_balance import add_paper_pnl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH  = PROJECT_ROOT / "data" / "trades.json"

# Default paper-trading fee — 0.05% per side (reasonable mid-tier crypto rate).
# Mirrors the Backtester's "futures" fee preset so paper PnL ≈ backtest PnL
# for the same signal stream.
DEFAULT_FEE_BPS = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_trades() -> list[dict]:
    raw = read_json(str(TRADES_PATH), default=[])
    if isinstance(raw, dict):
        raw = raw.get("trades", [])
    return list(raw or [])


def _save_trades(trades: list[dict]) -> None:
    write_json(str(TRADES_PATH), trades)


def book_market_order(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    *,
    strategy: str = "",
    market: str = "spot",
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict:
    """Record a paper market order. Returns a dict that mimics the CCXT
    order shape so the bot's existing handlers don't need branching."""
    cost = float(amount) * float(price)
    fee  = cost * (fee_bps / 10_000)
    order_id = f"paper-{uuid.uuid4().hex[:12]}"
    record = {
        "id":          order_id,
        "is_paper":    True,
        "symbol":      symbol,
        "side":        side.upper(),
        "amount":      float(amount),
        "price":       float(price),
        "cost":        round(cost, 4),
        "ts":          _now(),
        "buy_time":    _now() if side.upper() == "BUY"  else "",
        "sell_time":   _now() if side.upper() == "SELL" else "",
        "buy_price":   float(price) if side.upper() == "BUY"  else None,
        "sell_price":  float(price) if side.upper() == "SELL" else None,
        "fee":         {"cost": round(fee, 4), "currency": "USDT"},
        "info":        {"source": "paper_book"},
        "status":      "OPEN",
        "strategy":    strategy or "manual",
        "market":      market,
        "pnl_usdt":    0.0,
    }
    trades = _load_trades()
    trades.append(record)
    _save_trades(trades)
    logger.info("[paper] booked %s %s amt=%s price=%s fee=%.4f USDT (id=%s)",
                side.upper(), symbol, amount, price, fee, order_id)
    return record


def book_close(
    order_id: str,
    exit_price: float,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict | None:
    """Close a previously-opened paper trade. Computes net PnL after
    round-trip fees and credits the virtual balance via add_paper_pnl().
    Returns the updated trade record, or None if the order_id wasn't found."""
    trades = _load_trades()
    for t in trades:
        if t.get("id") == order_id and t.get("is_paper") and t.get("status") != "CLOSED":
            entry_price = float(t.get("buy_price") or t.get("price") or 0)
            amount = float(t.get("amount") or 0)
            exit_cost  = amount * float(exit_price)
            entry_cost = amount * entry_price
            side       = t.get("side", "BUY").upper()
            # Long PnL = (exit - entry) * amount; short = inverse
            gross = (exit_cost - entry_cost) if side == "BUY" else (entry_cost - exit_cost)
            entry_fee = entry_cost * (fee_bps / 10_000)
            exit_fee  = exit_cost  * (fee_bps / 10_000)
            net = gross - entry_fee - exit_fee
            t["status"]     = "CLOSED"
            t["sell_price"] = float(exit_price)
            t["sell_time"]  = _now()
            t["pnl_usdt"]   = round(net, 4)
            t["gross_pnl"]  = round(gross, 4)
            t["total_fees_usdt"] = round(entry_fee + exit_fee, 4)
            _save_trades(trades)
            try:
                add_paper_pnl(net)
            except Exception as exc:
                logger.warning("[paper] virtual-balance update failed: %s", exc)
            logger.info("[paper] closed %s id=%s exit=%s net_pnl=%.4f USDT",
                        t.get("symbol"), order_id, exit_price, net)
            return t
    logger.warning("[paper] book_close: id=%s not found or already closed", order_id)
    return None


__all__ = ["book_market_order", "book_close", "DEFAULT_FEE_BPS"]
