"""
Realtime DB Writer — Phase 7.

Streams Binance kline events over WebSocket and writes them to QuestDB
(hot path) on the fly. Optionally rolls QuestDB data into the Parquet cold
store on a nightly schedule.

Architecture:
    Binance WS  --> kline_closed events  --> QuestDB ILP (port 9009)
                                          --> on-tick callback for the bot
                                          --> nightly: QuestDB --> Parquet

Run:
    python -m src.data_ingestion.realtime_db_writer
    python -m src.data_ingestion.realtime_db_writer --symbols BTC/USDT ETH/USDT --tfs 1m 1h
    python -m src.data_ingestion.realtime_db_writer --no-cold-rollover

The writer is **idempotent on restart**: it only writes bars whose
`closed=True` flag is set, and the QuestDB schema's primary key
(symbol+timeframe+timestamp) prevents duplicates.

This works alongside `src/data_ingestion/orderbook_collector.py` which
handles L2 snapshots. Different streams, different DB tables — no overlap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Phase 2 of the QuestDB → ParquetClient migration: callers keep the
# `get_questdb()` name for now; the underlying client is the file-based
# DuckDB+Parquet store. Renamed to `get_db_client` in Phase 5.
from src.database.parquet_client import get_client as get_questdb

logger = logging.getLogger("realtime_db")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
DEFAULT_TFS     = ["1m", "5m", "1h", "1d"]


def _stream_name(symbol: str, timeframe: str) -> str:
    """`BTC/USDT` + `1m` → `btcusdt@kline_1m`"""
    return f"{symbol.replace('/', '').lower()}@kline_{timeframe}"


def _ws_url(symbols: Iterable[str], timeframes: Iterable[str]) -> str:
    streams = "/".join(_stream_name(s, t) for s in symbols for t in timeframes)
    return f"{BINANCE_WS_BASE}?streams={streams}"


def _parse_kline_event(event: dict) -> dict | None:
    """Binance kline_<tf> event → bar dict suitable for write_market_candle.

    Only emits *closed* bars (`k.x == True`); in-progress bars are ignored
    so QuestDB never holds partial candles.
    """
    data = event.get("data") or {}
    k = data.get("k")
    if not k or not k.get("x"):  # x = is this kline closed?
        return None
    stream = event.get("stream", "")
    parts = stream.split("@")
    if len(parts) < 2 or not parts[1].startswith("kline_"):
        return None
    timeframe = parts[1].removeprefix("kline_")
    bare = parts[0]                       # btcusdt
    symbol = (bare[:-4] + "/" + bare[-4:]).upper() if bare.endswith("usdt") else bare.upper()
    return {
        "symbol":    symbol,
        "timeframe": timeframe,
        "timestamp": int(k["t"]),         # open time, ms
        "close_time": int(k["T"]),        # close time, ms
        "open":      float(k["o"]),
        "high":      float(k["h"]),
        "low":       float(k["l"]),
        "close":     float(k["c"]),
        "volume":    float(k["v"]),
        "trades_count": int(k.get("n") or 0),
        "taker_buy_base":  float(k.get("V") or 0),
        "taker_buy_quote": float(k.get("Q") or 0),
    }


_STATUS_PATH = PROJECT_ROOT / "data" / "realtime_status.json"


def _write_status(*, connected: bool, symbol: str = "--", n_written: int = 0,
                  last_msg_iso: str = "", error: str = "") -> None:
    """Heartbeat file consumed by the dashboard's component-health probe."""
    try:
        _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATUS_PATH.write_text(json.dumps({
            "connected": bool(connected),
            "symbol": symbol,
            "n_written": int(n_written),
            "last_msg_iso": last_msg_iso,
            "last_update_ts": time.time(),
            "error": error,
        }))
    except Exception as exc:
        logger.debug("realtime status write failed: %s", exc)


async def stream_loop(
    symbols: list[str],
    timeframes: list[str],
    *,
    on_bar=None,
) -> None:
    """Main WS loop. Auto-reconnects with exponential backoff."""
    try:
        import websockets
    except ImportError:
        logger.error("`websockets` not installed.")
        _write_status(connected=False, error="websockets not installed")
        return

    qdb = get_questdb()
    if not qdb.is_available():
        logger.warning("QuestDB unreachable at startup -- will still try writes per-tick.")

    url = _ws_url(symbols, timeframes)
    logger.info("[realtime] Connecting %d symbols x %d timeframes",
                len(symbols), len(timeframes))
    _write_status(connected=False, symbol=",".join(symbols[:3]))

    backoff = 1.0
    n_written = 0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                logger.info("[realtime] Connected. Streams: %s", _ws_url(symbols, timeframes))
                backoff = 1.0
                _write_status(connected=True, symbol=",".join(symbols[:3]),
                              n_written=n_written,
                              last_msg_iso=datetime.now(timezone.utc).isoformat())
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)
                    bar = _parse_kline_event(event)
                    if bar is None:
                        continue
                    ok = qdb.write_market_candle(bar["symbol"], bar["timeframe"], bar)
                    if ok:
                        n_written += 1
                        if n_written % 100 == 0:
                            logger.info("[realtime] %d bars written", n_written)
                            _write_status(connected=True, symbol=bar["symbol"],
                                          n_written=n_written,
                                          last_msg_iso=datetime.now(timezone.utc).isoformat())
                    if on_bar:
                        try:
                            on_bar(bar)
                        except Exception as exc:
                            logger.warning("[realtime] on_bar callback raised: %s", exc)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("[realtime] Stopped.")
            _write_status(connected=False, symbol=",".join(symbols[:3]),
                          n_written=n_written, error="stopped")
            return
        except Exception as exc:
            logger.warning("[realtime] WS error: %s -- reconnecting in %.1fs", exc, backoff)
            _write_status(connected=False, symbol=",".join(symbols[:3]),
                          n_written=n_written, error=str(exc)[:200])
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


# ─── Cold-rollover: QuestDB → Parquet ────────────────────────────────────────

async def cold_rollover_loop(symbols: list[str], timeframes: list[str],
                             *, interval_hours: float = 24.0) -> None:
    """Background task: every N hours, export the previous day's QuestDB rows
    into the Parquet store and (idempotently) keep both copies until QuestDB's
    retention policy aged them out.
    """
    from src.database.parquet_store import get_store
    qdb = get_questdb()
    parquet = get_store()

    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            d_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            d_end   = d_start + timedelta(days=1)

            for sym in symbols:
                for tf in timeframes:
                    # Phase A5 (2026-05-12): parameterized — sym/tf
                    # originate from watchlist JSON which goes through
                    # the auth-gated API but is still untrusted as a
                    # data-plane value.
                    sql = ("SELECT timestamp, open, high, low, close, volume "
                           "FROM market_data "
                           "WHERE symbol = ? AND timeframe = ? "
                           "  AND timestamp >= ? "
                           "  AND timestamp <  ? "
                           "ORDER BY timestamp")
                    try:
                        df = qdb.query_df(sql, params=[
                            sym, tf,
                            d_start.isoformat(),
                            d_end.isoformat(),
                        ])
                    except Exception as exc:
                        logger.warning("[rollover] %s/%s query failed: %s", sym, tf, exc)
                        continue
                    if df is None or df.empty:
                        continue
                    # Append into the parquet partition. We piggy-back on
                    # ParquetStore.ingest_csv via a temp file (simplest path
                    # that doesn't require a new ingest_dataframe API).
                    import tempfile, gzip
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv.gz", delete=False,
                        dir=str(PROJECT_ROOT / "data" / "cache"),
                    ) as tmp:
                        with gzip.open(tmp.name, "wt", encoding="utf-8", newline="") as f:
                            df.to_csv(f, index=False)
                        try:
                            parquet.ingest_csv(tmp.name, sym, timeframe=tf)
                            logger.info("[rollover] %s/%s -> parquet (%d rows)",
                                        sym, tf, len(df))
                        finally:
                            try:
                                Path(tmp.name).unlink()
                            except OSError:
                                pass
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("[rollover] stopped")
            return
        except Exception as exc:
            logger.warning("[rollover] error: %s", exc)


# ─── Main ────────────────────────────────────────────────────────────────────

async def heartbeat_loop(symbols: list[str], interval_s: float = 30.0) -> None:
    """Refresh `data/realtime_status.json` every N seconds so the dashboard
    sees a fresh heartbeat even on quiet bars (e.g. low-volume symbols
    between closed klines)."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            try:
                cur = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
            _write_status(
                connected=bool(cur.get("connected", False)),
                symbol=cur.get("symbol", ",".join(symbols[:3])),
                n_written=int(cur.get("n_written", 0)),
                last_msg_iso=cur.get("last_msg_iso", ""),
                error=cur.get("error", ""),
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            return
        except Exception as exc:
            logger.debug("[heartbeat] tick failed: %s", exc)


async def _main_async(symbols: list[str], timeframes: list[str],
                      *, cold_rollover: bool, rollover_hours: float) -> int:
    tasks = [
        asyncio.create_task(stream_loop(symbols, timeframes)),
        asyncio.create_task(heartbeat_loop(symbols)),
    ]
    if cold_rollover:
        tasks.append(asyncio.create_task(
            cold_rollover_loop(symbols, timeframes, interval_hours=rollover_hours)
        ))
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        for t in tasks:
            t.cancel()
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser(description="Realtime Binance WS -> QuestDB writer")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--tfs", nargs="+", default=None,
                   help="Timeframes, e.g. 1m 5m 1h 1d (default 1m 5m 1h 1d)")
    p.add_argument("--no-cold-rollover", action="store_true",
                   help="Disable nightly QuestDB->Parquet rollover")
    p.add_argument("--rollover-hours", type=float, default=24.0)
    args = p.parse_args()

    syms = [s.upper() for s in (args.symbols or DEFAULT_SYMBOLS)]
    tfs  = args.tfs or DEFAULT_TFS

    return asyncio.run(_main_async(
        syms, tfs,
        cold_rollover=not args.no_cold_rollover,
        rollover_hours=args.rollover_hours,
    ))


if __name__ == "__main__":
    sys.exit(main())
