"""
DatabaseAgent — subscribes to the agent bus and persists ALL bot data to
the file-based ParquetClient store (DuckDB + partitioned Parquet on D:).

Subscribes to:
  candle         → hot.market_data
  signal         → hot.model_signals
  order / trade  → cold.trade_events
  strategy_pnl   → cold.strategy_performance
  training_event → hot.training_telemetry
  news           → hot.news_sentiment
  heartbeat      → hot.agent_heartbeats

Also persists agent heartbeats every HEARTBEAT_SEC and strategy stats every STATS_SEC.

Runs as a background BaseAgent — add to agent bus alongside other agents.
Falls back silently when ParquetClient is unavailable (no data loss in main pipeline).

Originally targeted QuestDB; cut over in commits 43db156…b64b733. The ILP-format
emission stays — ParquetClient.write_ilp() parses it via its compat shim. The
three helpers (_to_ns / _tag / _now_ns) are inlined at module scope so this file
no longer depends on the (retired) src/database/questdb_client.py shim.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEARTBEAT_SEC = 30     # write agent heartbeats every N seconds
STATS_SEC     = 60     # write strategy stats every N seconds
BUFFER_LIMIT  = 2000   # max queued rows before dropping (back-pressure)


# Inlined ILP-format helpers (formerly in questdb_client.py — that file is
# being retired in this commit). ParquetClient.write_ilp() parses the lines
# emitted using these helpers via its built-in compatibility shim.
def _to_ns(ts) -> int | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1e9)
    if isinstance(ts, (int, float)):
        if ts < 1e12:
            return int(ts * 1e9)
        elif ts < 1e15:
            return int(ts * 1e6)
        return int(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S.%f+00:00"):
            try:
                dt = datetime.strptime(ts.replace("+00:00", ""),
                                       fmt.replace("+00:00", ""))
                return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
            except ValueError:
                continue
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1e9)
    return None


def _tag(v: Any) -> str:
    return (str(v).replace(" ", "_")
                  .replace(",", "_")
                  .replace("=", "_")
                  .replace("/", "_"))


def _now_ns() -> int:
    return int(time.time() * 1e9)


class DatabaseAgent:
    """
    Lightweight DB writer — not a full BaseAgent; runs its own daemon thread.
    Call start() once; it will flush buffers and write heartbeats continuously.
    """

    NAME = "DatabaseAgent"

    def __init__(self, bus=None):
        self._bus = bus
        self._client = None          # lazy init
        self._lock = threading.Lock()
        self._running = False

        # Per-topic queues
        self._candle_buf:   deque[tuple] = deque(maxlen=BUFFER_LIMIT)
        self._signal_buf:   deque[dict]  = deque(maxlen=BUFFER_LIMIT)
        self._trade_buf:    deque[dict]  = deque(maxlen=BUFFER_LIMIT)
        self._news_buf:     deque[dict]  = deque(maxlen=BUFFER_LIMIT)

        self._last_stats_write  = 0.0
        self._last_hb_write     = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._bus:
            self._bus.subscribe("candle",         self._on_candle)
            self._bus.subscribe("sim_candle",     self._on_candle)
            self._bus.subscribe("signal",         self._on_signal)
            self._bus.subscribe("order",          self._on_trade)
            self._bus.subscribe("trade",          self._on_trade)
            self._bus.subscribe("strategy_pnl",   self._on_strategy_pnl)
            self._bus.subscribe("training_event", self._on_training_event)
            self._bus.subscribe("news",           self._on_news)
        threading.Thread(target=self._flush_loop, daemon=True, name="db-agent-flush").start()
        logger.info("[DB] DatabaseAgent started")

    def stop(self) -> None:
        self._running = False

    # ── Bus callbacks (fast — just queue) ─────────────────────────────────────

    def _on_candle(self, msg) -> None:
        bar = msg.payload
        sym = bar.get("symbol", "BTC_USDT")
        tf  = bar.get("timeframe", "1m")
        self._candle_buf.append((sym, tf, bar))

    def _on_signal(self, msg) -> None:
        d = msg.payload or {}
        self._signal_buf.append(d)

    def _on_trade(self, msg) -> None:
        d = msg.payload or {}
        self._trade_buf.append(d)

    def _on_strategy_pnl(self, msg) -> None:
        d = msg.payload or {}
        self._trade_buf.append({
            "symbol":   d.get("symbol"),
            "strategy": d.get("strategy"),
            "pnl_usd":  d.get("pnl", 0),
            "size_usd": 0,
            "direction": 0,
            "is_live":  False,
        })

    def _on_training_event(self, msg) -> None:
        d = msg.payload or {}
        c = self._client_safe()
        if c:
            c.write_training_event(
                d.get("model_name", "unknown"),
                d.get("run_id", ""),
                d,
            )

    def _on_news(self, msg) -> None:
        d = msg.payload or {}
        self._news_buf.append(d)

    # ── Flush loop ────────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while self._running:
            try:
                self._flush_all()
            except Exception as exc:
                logger.debug("[DB] flush error: %s", exc)
            time.sleep(5)

    def _flush_all(self) -> None:
        c = self._client_safe()
        if not c:
            return

        # Candles
        if self._candle_buf:
            batch: list[tuple] = []
            while self._candle_buf:
                batch.append(self._candle_buf.popleft())
            if batch:
                lines = []
                for sym, tf, bar in batch:
                    ts_ns = _to_ns(bar.get("timestamp") or bar.get("ts"))
                    if ts_ns is None:
                        continue
                    lines.append(
                        f"market_data,symbol={_tag(sym)},timeframe={_tag(tf)} "
                        f"open={float(bar.get('open', 0))},"
                        f"high={float(bar.get('high', 0))},"
                        f"low={float(bar.get('low', 0))},"
                        f"close={float(bar.get('close', 0))},"
                        f"volume={float(bar.get('volume', 0))},"
                        f"funding_rate={float(bar.get('funding_rate') or 0)} "
                        f"{ts_ns}"
                    )
                if lines:
                    c.write_ilp(lines)

        # Signals
        if self._signal_buf:
            batch = []
            while self._signal_buf:
                batch.append(self._signal_buf.popleft())
            for sig in batch:
                sym = sig.pop("symbol", "UNKNOWN")
                ts  = sig.pop("ts", None) or sig.pop("timestamp", None)
                c.write_signal(sym, sig, ts)

        # Trades
        if self._trade_buf:
            batch = []
            while self._trade_buf:
                batch.append(self._trade_buf.popleft())
            lines = []
            for t in batch:
                ts_ns = _to_ns(t.get("timestamp")) or _now_ns()
                sym  = _tag(t.get("symbol", "UNKNOWN"))
                strat = _tag(t.get("strategy", "unknown"))
                mkt  = _tag(t.get("market", "FUTURES"))
                lines.append(
                    f"trade_events,symbol={sym},strategy={strat},market={mkt} "
                    f"direction={int(t.get('direction', 0))}i,"
                    f"entry_price={float(t.get('entry_price', 0))},"
                    f"exit_price={float(t.get('exit_price', 0))},"
                    f"size_usd={float(t.get('size_usd', 0))},"
                    f"pnl_usd={float(t.get('pnl_usd', 0))},"
                    f"fees_usd={float(t.get('fees_usd', 0))},"
                    f"bars_held={int(t.get('bars_held', 0))}i,"
                    f"is_live={str(t.get('is_live', False)).lower()} "
                    f"{ts_ns}"
                )
            if lines:
                c.write_ilp(lines)

        # News
        if self._news_buf:
            batch = []
            while self._news_buf:
                batch.append(self._news_buf.popleft())
            for item in batch:
                c.write_news_sentiment(item)

        # Strategy stats (periodic)
        now = time.monotonic()
        if now - self._last_stats_write >= STATS_SEC:
            self._write_strategy_stats(c)
            self._last_stats_write = now

        # Agent heartbeats (periodic)
        if now - self._last_hb_write >= HEARTBEAT_SEC:
            self._write_heartbeats(c)
            self._last_hb_write = now

    def _write_strategy_stats(self, c) -> None:
        try:
            from src.engine.agents.strategy_simulator import StrategySimulatorAgent
            # Try to find running instance
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, StrategySimulatorAgent):
                    c.write_strategy_stats(obj.get_stats())
                    return
        except Exception as exc:
            logger.debug("[DB] strategy stats write: %s", exc)

    def _write_heartbeats(self, c) -> None:
        try:
            import json as _j
            status_file = PROJECT_ROOT / "data" / "agent_status.json"
            if not status_file.exists():
                return
            data = _j.loads(status_file.read_text(encoding="utf-8"))
            lines = []
            ts_ns = _now_ns()
            for agent_name, info in data.items():
                status = _tag(info.get("status", "idle"))
                task   = str(info.get("current_task", ""))[:80].replace('"', '\\"')
                lines.append(
                    f'agent_heartbeats,agent={_tag(agent_name)},status={status} '
                    f'current_task="{task}" '
                    f'{ts_ns}'
                )
            if lines:
                c.write_ilp(lines)
        except Exception as exc:
            logger.debug("[DB] heartbeat write: %s", exc)

    # ── Helper ────────────────────────────────────────────────────────────────

    def _client_safe(self):
        if self._client is None:
            try:
                from src.database.parquet_client import get_client
                self._client = get_client()
            except Exception:
                return None
        if not self._client.is_available():
            return None
        return self._client
