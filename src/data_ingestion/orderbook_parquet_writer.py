"""
L2 orderbook → Parquet writer.

The 2026-05-13 X1.2 audit found the gap: `orderbook_collector` already runs
and publishes L2 aggregates on the ZeroMQ orderflow channel, but nothing
PERSISTS them to disk. So training pipelines that call
`feature_engineering.add_orderbook_features` find the `p_bid`/`p_ask`/
`v_bid`/`v_ask` columns absent (the function no-ops) and never learn from L2.

This module closes the loop:
  - Subscribe to DataBus orderflow channel.
  - Batch snapshots in memory (default: every 1000 rows OR 30 seconds).
  - Flush each batch to `data/parquet/_L2/<SYMBOL>/yyyymm=<YYYYMM>/<ts>.parquet`.
  - DuckDB / ParquetClient queries can now LEFT JOIN L2 onto candles at
    training time so add_orderbook_features actually has columns to work with.

Storage schema per row:
  - ts (int64, milliseconds since epoch — matches the rest of the store)
  - symbol (string, e.g. "BTC_USDT" — bot internal format)
  - p_bid (float64): best bid price
  - p_ask (float64): best ask price
  - v_bid (float64): aggregated top-N bid volume
  - v_ask (float64): aggregated top-N ask volume
  - depth (int32): how many levels were aggregated (collector param)

Run:
    python -m src.data_ingestion.orderbook_parquet_writer

Will claim the 'orderbook_writer' process-registry role so duplicates are
blocked at startup.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("orderbook_parquet_writer")

L2_PARQUET_DIR = PROJECT_ROOT / 'data' / 'parquet' / '_L2'
DEFAULT_BATCH_SIZE = 1000
DEFAULT_FLUSH_SEC = 30.0


def _normalize_symbol(symbol: str) -> str:
    """Collector emits 'BTC/USDT'; bot's Parquet store uses 'BTC_USDT'."""
    return symbol.replace('/', '_').upper()


def _partition_path(symbol: str, ts_ms: int) -> Path:
    """Path: data/parquet/_L2/<SYMBOL>/yyyymm=<YYYYMM>/<ts>.parquet"""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    yyyymm = dt.strftime('%Y%m')
    return L2_PARQUET_DIR / symbol / f'yyyymm={yyyymm}'


class OrderbookParquetWriter:
    """Batches L2 snapshots and writes them as Parquet partitions."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE,
                 flush_sec: float = DEFAULT_FLUSH_SEC):
        self.batch_size = batch_size
        self.flush_sec = flush_sec
        self._buf: list[dict] = []
        self._buf_lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._stop = False
        L2_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    def on_snapshot(self, snap: dict) -> None:
        """Bus callback — append to in-memory buffer; trigger flush if full."""
        row = self._snap_to_row(snap)
        if row is None:
            return
        with self._buf_lock:
            self._buf.append(row)
            should_flush = (
                len(self._buf) >= self.batch_size
                or (time.monotonic() - self._last_flush) >= self.flush_sec
            )
        if should_flush:
            self.flush()

    @staticmethod
    def _snap_to_row(snap: dict) -> dict | None:
        """Convert one DataBus snapshot to a flat row."""
        try:
            symbol = _normalize_symbol(snap.get('symbol', ''))
            if not symbol:
                return None
            ts_ms = int(snap.get('timestamp') or 0)
            if ts_ms <= 0:
                return None
            # aggregate_levels emits keys p_bid/p_ask/v_bid/v_ask; defensive cast.
            return {
                'ts':     ts_ms,
                'symbol': symbol,
                'p_bid':  float(snap.get('p_bid') or 0.0),
                'p_ask':  float(snap.get('p_ask') or 0.0),
                'v_bid':  float(snap.get('v_bid') or 0.0),
                'v_ask':  float(snap.get('v_ask') or 0.0),
                'depth':  int(snap.get('depth') or 0),
            }
        except Exception as exc:
            logger.debug("[L2 writer] bad snapshot dropped: %s", exc)
            return None

    def flush(self) -> int:
        """Drain the buffer to a partitioned Parquet file. Returns rows written.

        Idempotent on empty buffer (returns 0, no side effect). Safe to call
        from any thread — uses an internal lock to swap the buffer atomically.
        """
        with self._buf_lock:
            if not self._buf:
                return 0
            rows = self._buf
            self._buf = []
            self._last_flush = time.monotonic()
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            # Group by partition path so a 30-second batch spanning a month
            # boundary writes into TWO files instead of conflating them.
            for (sym, part_path), grp in df.groupby(
                    [df['symbol'], df['ts'].map(
                        lambda t: _partition_path(_normalize_symbol(rows[0]['symbol']), t)
                    )],
            ):
                # The groupby key for the path lambda re-derives by row; this
                # is correct but verbose — replaced below by a direct path.
                pass
            written = 0
            for sym in df['symbol'].unique():
                sub = df[df['symbol'] == sym]
                # Within one symbol, split by yyyymm in case the batch spans
                # a month boundary (rare but possible).
                sub = sub.assign(_yyyymm=sub['ts'].map(
                    lambda t: datetime.fromtimestamp(t/1000.0, tz=timezone.utc).strftime('%Y%m')
                ))
                for yyyymm, mgrp in sub.groupby('_yyyymm'):
                    out_dir = L2_PARQUET_DIR / sym / f'yyyymm={yyyymm}'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = out_dir / f'l2_{int(time.time()*1000)}.parquet'
                    mgrp.drop(columns=['_yyyymm']).to_parquet(fname, index=False)
                    written += len(mgrp)
            logger.info("[L2 writer] flushed %d rows", written)
            return written
        except Exception as exc:
            # On flush failure put the rows back so we don't lose them.
            logger.error("[L2 writer] flush failed: %s — re-buffering %d rows",
                         exc, len(rows))
            with self._buf_lock:
                self._buf = rows + self._buf
            return 0

    def stop(self) -> None:
        """Final drain on shutdown."""
        self._stop = True
        self.flush()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--flush-sec', type=float, default=DEFAULT_FLUSH_SEC)
    args = parser.parse_args()

    # Singleton — only one writer per host.
    try:
        from src.utils.process_registry import claim_role, release_role, heartbeat
        import atexit
        ok, existing = claim_role('orderbook_writer', by='src.data_ingestion.orderbook_parquet_writer')
        if not ok:
            logger.error(
                "Another orderbook_writer already running: PID=%s. Exiting.",
                existing.get('pid'),
            )
            sys.exit(0)
        atexit.register(lambda: release_role('orderbook_writer', reason='atexit'))
        def _hb_loop():
            while True:
                time.sleep(60)
                try: heartbeat('orderbook_writer')
                except Exception: pass
        threading.Thread(target=_hb_loop, daemon=True, name='registry-hb').start()
    except Exception as exc:
        logger.warning("[startup] process_registry unavailable: %s", exc)

    writer = OrderbookParquetWriter(
        batch_size=args.batch_size, flush_sec=args.flush_sec,
    )

    from src.transport.data_bus import get_data_bus
    bus = get_data_bus()
    bus.subscribe_orderflow(writer.on_snapshot)
    logger.info("[L2 writer] subscribed to orderflow bus; batch=%d, flush_sec=%.1f",
                args.batch_size, args.flush_sec)

    # Periodic time-based flush so trickle traffic doesn't sit in memory.
    def _periodic_flush():
        while not writer._stop:
            time.sleep(args.flush_sec)
            writer.flush()
    threading.Thread(target=_periodic_flush, daemon=True, name='l2-periodic-flush').start()

    # Graceful Ctrl+C
    def _handler(_signum, _frame):
        logger.info("[L2 writer] shutting down; final flush…")
        writer.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # Block forever
    while not writer._stop:
        time.sleep(1)


if __name__ == '__main__':
    main()
