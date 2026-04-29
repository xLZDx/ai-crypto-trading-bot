"""
StrategySimulatorAgent — paper-trades all rule-based strategies on simulator candles.

Each strategy starts with VIRTUAL_CAPITAL virtual USD.  Signals are recomputed in
_run_cycle() every INTERVAL_SEC seconds using the same _build_signals() function as
the backtester, ensuring results are comparable.  The agent keeps all state in-memory
for speed; it writes summaries to SimulatorDataStore every PERSIST_EVERY cycles.

Publishes 'strategy_pnl' on AgentBus when positions change.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIRTUAL_CAPITAL   = 10_000.0   # per-strategy starting balance (USD)
BUFFER_SIZE       = 500        # rolling bar window per symbol
POSITION_SIZE_PCT = 0.10       # 10% of current balance per trade
FEE_RATE          = 0.0004     # 0.04% taker fee
MAX_HOLD_BARS     = 48         # auto-close after N bars
INTERVAL_SEC      = 5.0        # how often to run signal computation
PERSIST_EVERY     = 12         # persist to DB every N cycles (every ~1 min)

# Rule-based strategies and their signal column names
_STRATEGIES: dict[str, str] = {
    "RSI_MeanReversion":  "signal_rsi",
    "MACD_Momentum":      "signal_macd",
    "BB_Reversion":       "signal_bb",
    "Ensemble_A":         "signal_ensemble",
    "VWAP_Reversion":     "signal_vwap",
    "Donchian_Breakout":  "signal_donchian",
    "Keltner_Breakout":   "signal_keltner",
    "Funding_Arb":        "signal_funding",
    "Volatility_Breakout": "signal_vol_breakout",
    "SuperTrend":         "signal_supertrend",
}


class _Account:
    """Tracks one strategy's virtual trading account."""

    __slots__ = (
        "balance", "n_trades", "n_wins", "n_losses",
        "total_pnl", "position", "bars_in_position",
    )

    def __init__(self):
        self.balance:          float  = VIRTUAL_CAPITAL
        self.n_trades:         int    = 0
        self.n_wins:           int    = 0
        self.n_losses:         int    = 0
        self.total_pnl:        float  = 0.0
        self.position:         dict | None = None   # {direction, entry_price, size}
        self.bars_in_position: int    = 0

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins  = self.n_wins  * (self.total_pnl / max(self.n_trades, 1))
        loss  = self.n_losses * abs(self.total_pnl / max(self.n_trades, 1))
        return wins / loss if loss > 0 else 1.0

    def open_position(self, direction: int, price: float) -> None:
        size = self.balance * POSITION_SIZE_PCT
        self.position = {
            "direction":   direction,
            "entry_price": price,
            "size":        size,
            "entry_fee":   size * FEE_RATE,
        }
        self.bars_in_position = 0

    def close_position(self, price: float) -> float:
        if self.position is None:
            return 0.0
        pos    = self.position
        d      = pos["direction"]
        raw    = (price - pos["entry_price"]) / pos["entry_price"] * d * pos["size"]
        fees   = pos["entry_fee"] + pos["size"] * FEE_RATE
        pnl    = raw - fees
        self.balance   += pnl
        self.total_pnl += pnl
        self.n_trades  += 1
        if pnl > 0:
            self.n_wins += 1
        else:
            self.n_losses += 1
        self.position = None
        self.bars_in_position = 0
        return pnl

    def as_dict(self, strategy: str, symbol: str) -> dict:
        return {
            "strategy":      strategy,
            "symbol":        symbol,
            "balance":       round(self.balance, 2),
            "total_pnl":     round(self.total_pnl, 2),
            "pnl_pct":       round((self.balance / VIRTUAL_CAPITAL - 1) * 100, 2),
            "n_trades":      self.n_trades,
            "n_wins":        self.n_wins,
            "n_losses":      self.n_losses,
            "win_rate":      round(self.win_rate * 100, 1),
            "in_position":   self.position is not None,
            "position_dir":  self.position["direction"] if self.position else 0,
        }


class StrategySimulatorAgent(BaseAgent):
    """
    Paper-trades all registered rule-based strategies on live simulator feed.

    Architecture:
      _on_sim_candle: buffer bars per symbol (lock-free deque)
      _run_cycle:     every INTERVAL_SEC — compute signals, update positions
    """

    NAME = "StrategySimulatorAgent"

    def __init__(self, bus=None):
        super().__init__(bus=bus, interval_sec=INTERVAL_SEC)
        # Rolling bar buffer: symbol → deque[dict]
        self._bars: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))
        self._buf_lock = threading.Lock()

        # Per-strategy, per-symbol virtual accounts
        # accounts[strategy][symbol] = _Account
        self._accounts: dict[str, dict[str, _Account]] = {
            s: defaultdict(_Account) for s in _STRATEGIES
        }
        self._acct_lock = threading.Lock()

        # Recent bars counter (how many new bars since last cycle)
        self._new_bars: dict[str, int] = defaultdict(int)

        # Cycle counter for DB persistence
        self._cycle_count = 0
        self._store = None   # lazy-init

        # Stats cache for fast API reads
        self._stats_cache: list[dict] = []
        self._stats_ts: float = 0.0

    # ── AgentBus subscriptions ────────────────────────────────────────────────

    def _setup_subscriptions(self) -> None:
        self.bus.subscribe("sim_candle", self._on_sim_candle)

    def _on_sim_candle(self, msg) -> None:
        bar: dict = msg.payload
        symbol = bar.get("symbol", "BTC_USDT")
        with self._buf_lock:
            self._bars[symbol].append(bar)
            self._new_bars[symbol] += 1

    # ── BaseAgent cycle ───────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        with self._buf_lock:
            symbols_with_bars = {s: n for s, n in self._new_bars.items() if n >= 5}
            if symbols_with_bars:
                self._new_bars.update({s: 0 for s in symbols_with_bars})

        for symbol, new_count in symbols_with_bars.items():
            self._process_symbol(symbol)

        self._cycle_count += 1
        if self._cycle_count % PERSIST_EVERY == 0:
            self._persist_stats()

    # ── Signal computation + position management ──────────────────────────────

    def _process_symbol(self, symbol: str) -> None:
        with self._buf_lock:
            bars = list(self._bars[symbol])

        if len(bars) < 30:
            return

        # Build DataFrame
        try:
            df = pd.DataFrame(bars)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.set_index("timestamp")
            for col in ("open", "high", "low", "close", "volume", "funding_rate"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"]).sort_index()
            if len(df) < 20:
                return
        except Exception as exc:
            logger.debug("[StratSim] DataFrame build error for %s: %s", symbol, exc)
            return

        # Compute signals (reuse backtester's _build_signals)
        try:
            from src.engine.backtester import _build_signals
            df = _build_signals(df)
        except Exception as exc:
            logger.debug("[StratSim] _build_signals error for %s: %s", symbol, exc)
            return

        # Get the last row's signals
        last = df.iloc[-1]
        price = float(last["close"])
        if price <= 0:
            return

        with self._acct_lock:
            for strategy, sig_col in _STRATEGIES.items():
                try:
                    signal = float(last.get(sig_col, 0.0) or 0.0)
                    acct = self._accounts[strategy][symbol]
                    self._update_account(acct, signal, price, strategy, symbol)
                except Exception as exc:
                    logger.debug("[StratSim] Account update error %s/%s: %s", strategy, symbol, exc)

    def _update_account(
        self, acct: _Account, signal: float, price: float, strategy: str, symbol: str
    ) -> None:
        # Tick bars-in-position counter
        if acct.position is not None:
            acct.bars_in_position += 1

        # Close: signal flip or max hold
        should_close = False
        if acct.position is not None:
            d = acct.position["direction"]
            if (d == 1 and signal < -0.1) or (d == -1 and signal > 0.1):
                should_close = True
            if acct.bars_in_position >= MAX_HOLD_BARS:
                should_close = True

        if should_close:
            pnl = acct.close_position(price)
            self.publish("strategy_pnl", {
                "strategy": strategy, "symbol": symbol,
                "pnl": round(pnl, 4),
                "balance": round(acct.balance, 2),
            })

        # Open: new signal
        if acct.position is None and abs(signal) > 0.1 and acct.balance > 0:
            direction = 1 if signal > 0 else -1
            acct.open_position(direction, price)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_stats(self) -> list[dict]:
        """Return per-strategy performance summary. Thread-safe."""
        now = time.monotonic()
        if now - self._stats_ts < 2.0 and self._stats_cache:
            return self._stats_cache

        rows: list[dict] = []
        with self._acct_lock:
            for strategy in _STRATEGIES:
                accounts = self._accounts[strategy]
                if not accounts:
                    # No candles seen yet — return zero row
                    rows.append({
                        "strategy": strategy,
                        "symbol": "—",
                        "balance": VIRTUAL_CAPITAL,
                        "total_pnl": 0.0,
                        "pnl_pct": 0.0,
                        "n_trades": 0,
                        "n_wins": 0,
                        "n_losses": 0,
                        "win_rate": 0.0,
                        "in_position": False,
                        "position_dir": 0,
                    })
                    continue
                # Aggregate across all symbols
                agg: dict[str, Any] = {
                    "strategy": strategy,
                    "symbol":   "+".join(sorted(accounts.keys())),
                    "balance":  round(sum(a.balance  for a in accounts.values()), 2),
                    "total_pnl": round(sum(a.total_pnl for a in accounts.values()), 2),
                    "pnl_pct":  0.0,
                    "n_trades": sum(a.n_trades  for a in accounts.values()),
                    "n_wins":   sum(a.n_wins    for a in accounts.values()),
                    "n_losses": sum(a.n_losses  for a in accounts.values()),
                    "win_rate": 0.0,
                    "in_position": any(a.position is not None for a in accounts.values()),
                    "position_dir": next((a.position["direction"] for a in accounts.values()
                                         if a.position is not None), 0),
                }
                base = VIRTUAL_CAPITAL * len(accounts)
                agg["pnl_pct"]  = round((agg["balance"] / base - 1) * 100, 2) if base else 0.0
                agg["win_rate"] = round(agg["n_wins"] / agg["n_trades"] * 100, 1) if agg["n_trades"] else 0.0
                rows.append(agg)

        # Sort by PNL descending
        rows.sort(key=lambda r: r["total_pnl"], reverse=True)
        self._stats_cache = rows
        self._stats_ts = now
        return rows

    def reset_stats(self) -> None:
        """Reset all virtual accounts to initial capital."""
        with self._acct_lock:
            for strategy in _STRATEGIES:
                self._accounts[strategy] = defaultdict(_Account)
        self._stats_cache = []
        logger.info("[StratSim] All accounts reset to $%.0f", VIRTUAL_CAPITAL)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_stats(self) -> None:
        try:
            if self._store is None:
                from src.simulation.data_store import SimulatorDataStore
                self._store = SimulatorDataStore()
            stats = self.get_stats()
            self._store.save_strategy_stats(stats)
        except Exception as exc:
            logger.debug("[StratSim] persist error: %s", exc)
