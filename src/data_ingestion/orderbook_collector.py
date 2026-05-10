"""
L2 order book real-time collector — Phase 1, Level 1.

Subscribes to Binance public depth streams, reduces each snapshot to top-N
aggregates (P_bid, P_ask, V_bid, V_ask) and publishes them on the
ZeroMQ DataBus orderflow PUB socket. Consumers (bot, dashboard, training
batch builder) subscribe independently — no broker required.

Why direct websockets instead of ccxt.pro:
    - ccxt.pro is paid; the open-source ccxt 4.x has only REST async support.
    - Binance's public depth stream needs no auth and is rock-solid.

Stream URL pattern:
    wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms

Run:
    python -m src.data_ingestion.orderbook_collector --symbols BTC/USDT,ETH/USDT
    python -m src.data_ingestion.orderbook_collector --symbols BTC/USDT --depth 10 --speed 100ms
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.orderbook_features import aggregate_levels
from src.transport.data_bus import get_data_bus

logger = logging.getLogger("orderbook_collector")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"


def _stream_name(symbol: str, depth: int, speed: str) -> str:
    """`BTC/USDT` + depth=20 + speed=100ms  →  `btcusdt@depth20@100ms`"""
    bare = symbol.replace("/", "").lower()
    return f"{bare}@depth{depth}@{speed}"


def _binance_url(symbols: Iterable[str], depth: int, speed: str) -> str:
    streams = "/".join(_stream_name(s, depth, speed) for s in symbols)
    return f"{BINANCE_WS_BASE}?streams={streams}"


def _parse_depth_event(event: dict, depth: int) -> dict:
    """Binance `<symbol>@depth<N>` event → uniform snapshot dict."""
    data = event.get("data") or {}
    stream = event.get("stream", "")
    # stream like `btcusdt@depth20@100ms` → symbol "BTC/USDT"
    bare = stream.split("@", 1)[0]
    symbol = (bare[:-4] + "/" + bare[-4:]).upper() if bare.endswith("usdt") else bare.upper()
    bids = data.get("bids") or data.get("b") or []
    asks = data.get("asks") or data.get("a") or []
    snapshot = {
        "symbol":    symbol,
        "timestamp": int(data.get("E") or datetime.now(timezone.utc).timestamp() * 1000),
        "bids":      [[float(p), float(q)] for p, q in bids],
        "asks":      [[float(p), float(q)] for p, q in asks],
    }
    return aggregate_levels(snapshot, depth=depth)


async def stream_loop(
    symbols: list[str],
    depth: int = 20,
    speed: str = "100ms",
    publish: bool = True,
    on_snapshot=None,
) -> None:
    """Connect, parse, publish in a loop. Auto-reconnect on disconnect."""
    try:
        import websockets
    except ImportError:
        logger.error("`websockets` package missing. pip install websockets")
        return

    bus = get_data_bus() if publish else None
    url = _binance_url(symbols, depth, speed)
    logger.info("[OB] Connecting %s", url)

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                logger.info("[OB] Connected. Subscribed: %s", symbols)
                backoff = 1.0
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)
                    snap = _parse_depth_event(event, depth=depth)
                    if not snap:
                        continue
                    if bus:
                        bus.publish_orderflow(snap)
                    if on_snapshot:
                        on_snapshot(snap)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("[OB] Stopped by user.")
            return
        except Exception as exc:
            logger.warning("[OB] Stream error: %s — reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="L2 order book streaming collector")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--speed", default="100ms", choices=["100ms", "1000ms"])
    parser.add_argument("--no-publish", action="store_true",
                        help="Only print, don't publish to ZMQ DataBus")
    args = parser.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    try:
        asyncio.run(stream_loop(
            symbols=syms,
            depth=args.depth,
            speed=args.speed,
            publish=not args.no_publish,
            on_snapshot=lambda s: logger.info(
                "[OB] %s I=%+.3f Pmicro=%.4f", s["symbol"], _i(s), _pm(s)
            ),
        ))
    except KeyboardInterrupt:
        pass
    return 0


def _i(s):
    """Quick imbalance calc for log line."""
    denom = s["v_bid"] + s["v_ask"]
    return (s["v_bid"] - s["v_ask"]) / denom if denom > 0 else 0.0


def _pm(s):
    """Quick microprice for log line."""
    denom = s["v_bid"] + s["v_ask"]
    if denom <= 0:
        return (s["p_bid"] + s["p_ask"]) / 2
    return (s["p_ask"] * s["v_bid"] + s["p_bid"] * s["v_ask"]) / denom


if __name__ == "__main__":
    sys.exit(main())
