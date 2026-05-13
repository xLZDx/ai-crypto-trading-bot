"""
Tests for src/data_ingestion/orderbook_parquet_writer.py — X1.2 follow-on.

Covers the writer's pure-Python contract:
  - snapshot → row conversion (happy path + malformed snapshots dropped)
  - flush writes a parquet file in the expected partition path
  - flush is idempotent on empty buffer
  - flush failure re-buffers rows (data not lost)
  - partition path matches data/parquet/_L2/<SYM>/yyyymm=<YYYYMM>/

The ZMQ bus subscription is NOT covered here — that's an integration test;
unit tests only need the snapshot→parquet contract.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_writer(tmp_path, monkeypatch):
    from src.data_ingestion import orderbook_parquet_writer as obw
    monkeypatch.setattr(obw, 'L2_PARQUET_DIR', tmp_path / '_L2')
    writer = obw.OrderbookParquetWriter(batch_size=3, flush_sec=999.0)
    return obw, writer, tmp_path / '_L2'


def test_snap_to_row_happy_path(isolated_writer):
    obw, _, _ = isolated_writer
    snap = {
        'symbol': 'BTC/USDT', 'timestamp': 1778693000000,
        'p_bid': 79000.5, 'p_ask': 79001.5,
        'v_bid': 12.3,    'v_ask': 8.7,    'depth': 20,
    }
    row = obw.OrderbookParquetWriter._snap_to_row(snap)
    assert row is not None
    assert row['symbol'] == 'BTC_USDT'   # normalized to bot internal format
    assert row['ts']     == 1778693000000
    assert row['p_bid']  == pytest.approx(79000.5)
    assert row['v_bid']  == pytest.approx(12.3)
    assert row['depth']  == 20


def test_snap_to_row_drops_malformed(isolated_writer):
    obw, _, _ = isolated_writer
    # Missing symbol
    assert obw.OrderbookParquetWriter._snap_to_row({'timestamp': 123}) is None
    # Missing timestamp
    assert obw.OrderbookParquetWriter._snap_to_row({'symbol': 'BTC/USDT'}) is None
    # Zero timestamp
    assert obw.OrderbookParquetWriter._snap_to_row({'symbol': 'BTC/USDT', 'timestamp': 0}) is None


def test_flush_writes_parquet(isolated_writer):
    obw, writer, l2_dir = isolated_writer
    now_ms = int(time.time() * 1000)
    for i in range(5):
        writer.on_snapshot({
            'symbol': 'BTC/USDT', 'timestamp': now_ms + i,
            'p_bid': 79000.0 + i, 'p_ask': 79001.0 + i,
            'v_bid': 10.0,        'v_ask': 9.0,    'depth': 20,
        })
    # batch_size=3 → first flush at i=2, plus residue in buffer; force final flush
    writer.flush()
    # Expect a parquet file at data/parquet/_L2/BTC_USDT/yyyymm=<YYYYMM>/
    yyyymm = datetime.fromtimestamp(now_ms/1000.0, tz=timezone.utc).strftime('%Y%m')
    part_dir = l2_dir / 'BTC_USDT' / f'yyyymm={yyyymm}'
    assert part_dir.exists(), f'partition dir not created: {part_dir}'
    files = list(part_dir.glob('l2_*.parquet'))
    assert files, 'no parquet files written'
    # Read it back; should have 5 rows total across however many files
    import pandas as pd
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    assert len(df) == 5
    assert set(df.columns) >= {'ts', 'symbol', 'p_bid', 'p_ask', 'v_bid', 'v_ask', 'depth'}
    assert (df['symbol'] == 'BTC_USDT').all()


def test_flush_empty_is_noop(isolated_writer):
    obw, writer, l2_dir = isolated_writer
    assert writer.flush() == 0
    # No directory created
    assert not l2_dir.exists() or not any(l2_dir.iterdir())


def test_flush_failure_rebuffers(isolated_writer, monkeypatch):
    """If parquet write blows up, rows must NOT be lost — they go back into
    the buffer for the next flush."""
    obw, writer, _ = isolated_writer
    writer.on_snapshot({
        'symbol': 'ETH/USDT', 'timestamp': int(time.time()*1000),
        'p_bid': 1, 'p_ask': 2, 'v_bid': 3, 'v_ask': 4, 'depth': 10,
    })
    # Force flush to raise
    import pandas as pd
    orig_to_parquet = pd.DataFrame.to_parquet
    def boom(self, *a, **kw):
        raise IOError('disk full')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', boom)

    written = writer.flush()
    assert written == 0
    assert len(writer._buf) == 1, 'row must be re-buffered on flush failure'

    # Recovery: restore to_parquet, flush again — row gets written.
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', orig_to_parquet)
    assert writer.flush() == 1


def test_batch_spans_month_boundary(isolated_writer):
    """A 30s batch crossing midnight on month-end must produce TWO files
    (one per yyyymm partition), not conflate them."""
    obw, writer, l2_dir = isolated_writer
    # Pick a known boundary: 2026-04-30 23:59:59 UTC and 2026-05-01 00:00:01 UTC
    t1_ms = int(datetime(2026, 4, 30, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
    t2_ms = int(datetime(2026, 5, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    writer.on_snapshot({'symbol': 'BTC/USDT', 'timestamp': t1_ms,
                         'p_bid': 1, 'p_ask': 2, 'v_bid': 3, 'v_ask': 4, 'depth': 10})
    writer.on_snapshot({'symbol': 'BTC/USDT', 'timestamp': t2_ms,
                         'p_bid': 5, 'p_ask': 6, 'v_bid': 7, 'v_ask': 8, 'depth': 10})
    writer.flush()

    part_apr = l2_dir / 'BTC_USDT' / 'yyyymm=202604'
    part_may = l2_dir / 'BTC_USDT' / 'yyyymm=202605'
    assert part_apr.exists()
    assert part_may.exists()


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
