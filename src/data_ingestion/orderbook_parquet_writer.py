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


# Reviewer fix 2026-05-13: bound the in-memory buffer so a persistent flush
# failure (disk full / permissions) cannot OOM the process. When the cap is
# hit we drop OLDEST rows and emit a CRITICAL log — sustained disk-full is
# now a visible operator alert, not a silent crash.
_MAX_BUF_ROWS = 100_000
_SNAP_FAILURE_ALERT_AT = 100   # escalate to ERROR after N consecutive parse fails


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
        # Reviewer-fix counter: consecutive _snap_to_row failures. A bus
        # schema change would silently null-drop every snapshot; this counter
        # escalates the first 100 to a single ERROR with the raw payload so
        # the operator can diagnose. Resets on the next successful parse.
        self._consec_snap_failures = 0
        L2_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    def on_snapshot(self, snap: dict) -> None:
        """Bus callback — append to in-memory buffer; trigger flush if full."""
        row = self._snap_to_row(snap, writer=self)
        if row is None:
            return
        self._consec_snap_failures = 0
        with self._buf_lock:
            self._buf.append(row)
            # Cap the buffer — drop oldest if a flush failure has left the
            # buffer growing unbounded. Reviewer fix 2026-05-13.
            if len(self._buf) > _MAX_BUF_ROWS:
                dropped = len(self._buf) - _MAX_BUF_ROWS
                self._buf = self._buf[-_MAX_BUF_ROWS:]
                logger.critical(
                    "[L2 writer] buffer overflow: dropped %d oldest rows "
                    "(buffer capped at %d). Underlying flush is failing — "
                    "check logs/orderbook_parquet_writer.log and disk space.",
                    dropped, _MAX_BUF_ROWS,
                )
            should_flush = (
                len(self._buf) >= self.batch_size
                or (time.monotonic() - self._last_flush) >= self.flush_sec
            )
        if should_flush:
            self.flush()

    @staticmethod
    def _snap_to_row(snap: dict, writer=None) -> dict | None:
        """Convert one DataBus snapshot to a flat row.

        Reviewer fix 2026-05-13: silent-debug drops escalated. Counts
        consecutive failures via the writer instance; first 100 are
        DEBUG (matches the old behavior for noisy bus moments), then a
        single ERROR with the raw payload so a schema change becomes
        operator-visible.
        """
        try:
            symbol = _normalize_symbol(snap.get('symbol', ''))
            if not symbol:
                raise ValueError('missing symbol')
            ts_ms = int(snap.get('timestamp') or 0)
            if ts_ms <= 0:
                raise ValueError(f'bad timestamp {ts_ms}')
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
            if writer is not None:
                writer._consec_snap_failures += 1
                n = writer._consec_snap_failures
                if n == _SNAP_FAILURE_ALERT_AT:
                    logger.error(
                        "[L2 writer] %d consecutive snapshot parse failures — "
                        "bus payload may have changed shape. Last error: %s. "
                        "Sample payload (first 200 chars): %s",
                        n, exc, str(snap)[:200],
                    )
                else:
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
            # Per-symbol per-yyyymm partitioning. A 30-second batch that spans
            # a month boundary (rare) lands in TWO files, never conflated.
            written = 0
            for sym in df['symbol'].unique():
                sub = df[df['symbol'] == sym]
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
            logger.error("[L2 writer] flush failed: %s — re-buffering %d rows",
                         exc, len(rows))
            with self._buf_lock:
                # Cap-aware re-buffer: drop oldest if we'd exceed the limit.
                merged = rows + self._buf
                if len(merged) > _MAX_BUF_ROWS:
                    dropped = len(merged) - _MAX_BUF_ROWS
                    merged = merged[-_MAX_BUF_ROWS:]
                    logger.critical(
                        "[L2 writer] re-buffer would exceed cap %d; dropping "
                        "%d oldest rows. Sustained disk failure likely.",
                        _MAX_BUF_ROWS, dropped,
                    )
                self._buf = merged
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
            consec = 0
            while True:
                time.sleep(60)
                try:
                    if not heartbeat('orderbook_writer'):
                        consec += 1
                        if consec >= 3:
                            logger.critical(
                                "[registry-hb] orderbook_writer lost role ownership 3x — "
                                "exiting so a clean restart can claim"
                            )
                            import os as _os
                            _os._exit(0)
                    else:
                        consec = 0
                except Exception as exc:
                    logger.warning("[registry-hb] heartbeat failed: %s", exc)
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
