"""L2 / microstructure feature loader for downstream trainers.

Reads the partitioned L2 parquet store written by
src/data_ingestion/orderbook_parquet_writer.py:

    data/parquet/_L2/<SYMBOL>/yyyymm=<YYYYMM>/l2_<ts>.parquet
    columns: ts (ms), symbol, p_bid, p_ask, v_bid, v_ask, depth

Joins those snapshots to a bar timeline (1m / 5m / 15m / 1h / 4h / 1d) and
emits engineered features the TFT trainer can ingest as additional input
channels:

    micro_price       — volume-weighted between bid and ask
    spread_bps        — (ask - bid) / mid * 1e4
    book_imbalance    — (v_bid - v_ask) / (v_bid + v_ask) ∈ [-1, 1]
    ofi_5             — order-flow imbalance over the last 5 snapshots
    micro_vs_close    — (micro_price - close) / close
    spread_p95_window — rolling 95th-pct of spread over the bar window

Two consumption patterns:

    1) `load_bar_aligned(...)` — given a bar DataFrame (with `ts` ms),
       enrich it inline. Used by the TFT trainer.
    2) `load_raw_window(symbol, since_ms, until_ms)` — return the raw L2
       rows, e.g. for ad-hoc EDA or for tick-level (non-aggregated) trainers.

Implementation notes
--------------------
- DuckDB does the heavy lifting (read_parquet glob + window functions).
  Sets temp_directory to the project's cache dir, matching every other
  DuckDB call in this project (see trading-bot-helper skill).
- Missing partitions / empty datasets do NOT raise — the loader returns
  zero-valued columns so the trainer can fall back to OHLCV-only training.
- Causal: all features for bar B use only L2 snapshots that arrived BEFORE
  B's close timestamp (strict `ts < bar_close_ms`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
L2_PARQUET_DIR = PROJECT_ROOT / 'data' / 'parquet' / '_L2'
DUCKDB_TEMP_DIR = PROJECT_ROOT / 'data' / 'cache' / 'duckdb_temp'

L2_FEATURE_COLUMNS = (
    "l2_micro_price",
    "l2_spread_bps",
    "l2_book_imbalance",
    "l2_ofi_5",
    "l2_micro_vs_close",
    "l2_spread_p95_window",
    "l2_snapshot_count",   # for sanity / drop-when-too-few
)


def _normalise_symbol(symbol: str) -> str:
    """Match the orderbook_parquet_writer's partition naming."""
    return symbol.replace('/', '_').upper()


def l2_partitions_exist(symbol: str) -> bool:
    """Cheap check before invoking DuckDB. Saves a process spawn when the
    L2 writer hasn't accumulated anything for this symbol yet."""
    sym = _normalise_symbol(symbol)
    sym_dir = L2_PARQUET_DIR / sym
    if not sym_dir.exists():
        return False
    try:
        for yyyymm_dir in sym_dir.iterdir():
            if yyyymm_dir.is_dir() and any(yyyymm_dir.glob('*.parquet')):
                return True
    except OSError:
        pass
    return False


def _duckdb_con():
    """Open a DuckDB connection with the project's standard temp_directory."""
    import duckdb
    DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR.as_posix()}'")
    return con


def load_raw_window(
    symbol: str,
    since_ms: int | None = None,
    until_ms: int | None = None,
    *,
    limit: int | None = None,
):
    """Return the raw L2 snapshot rows for `symbol` in [since_ms, until_ms].

    Returns a pandas DataFrame with columns: ts, symbol, p_bid, p_ask,
    v_bid, v_ask, depth. Empty DataFrame if no partitions match.
    """
    import pandas as pd
    sym = _normalise_symbol(symbol)
    if not l2_partitions_exist(sym):
        return pd.DataFrame(columns=['ts', 'symbol', 'p_bid', 'p_ask',
                                     'v_bid', 'v_ask', 'depth'])
    pattern = (L2_PARQUET_DIR / sym).as_posix() + '/yyyymm=*/l2_*.parquet'
    where: list[str] = []
    if since_ms is not None:
        where.append(f"ts >= {int(since_ms)}")
    if until_ms is not None:
        where.append(f"ts < {int(until_ms)}")
    where_clause = (' WHERE ' + ' AND '.join(where)) if where else ''
    limit_clause = f' LIMIT {int(limit)}' if limit and limit > 0 else ''
    sql = (f"SELECT ts, symbol, p_bid, p_ask, v_bid, v_ask, depth "
           f"FROM read_parquet('{pattern}', union_by_name=true)"
           f"{where_clause} ORDER BY ts ASC{limit_clause}")
    try:
        with _duckdb_con() as con:
            return con.execute(sql).fetchdf()
    except Exception as e:
        logger.warning("l2_feature_loader: raw window read failed: %s", e)
        return pd.DataFrame(columns=['ts', 'symbol', 'p_bid', 'p_ask',
                                     'v_bid', 'v_ask', 'depth'])


def load_bar_aligned(
    symbol: str,
    bar_ts_ms: Iterable[int],
    bar_size_ms: int,
    closes: Iterable[float] | None = None,
):
    """Enrich a bar timeline with L2 features.

    `bar_ts_ms`     : iterable of bar OPEN timestamps in ms, monotonic asc.
    `bar_size_ms`   : duration of one bar in ms (e.g. 60_000 for 1m).
    `closes`        : optional iterable of close prices aligned with bar_ts_ms;
                      used for `l2_micro_vs_close`. If None, that column is 0.

    Returns a pandas DataFrame of len(bar_ts_ms) rows, one column per
    `L2_FEATURE_COLUMNS`. Each row aggregates L2 snapshots whose
    `ts` falls in [bar_open, bar_close) — strictly causal w.r.t. bar close.
    Bars with no snapshots get zeros (and snapshot_count=0).
    """
    import pandas as pd
    bar_ts_ms = list(int(t) for t in bar_ts_ms)
    n = len(bar_ts_ms)
    closes_list = list(float(c) for c in (closes or [0.0] * n))
    out = pd.DataFrame({
        'ts': bar_ts_ms,
        'close': closes_list[:n] + [0.0] * max(0, n - len(closes_list)),
    })
    for col in L2_FEATURE_COLUMNS:
        out[col] = 0.0
    sym = _normalise_symbol(symbol)
    if n == 0 or not l2_partitions_exist(sym):
        return out
    # Single window query: pull all L2 in [first_open, last_close), then
    # bucket in pandas — faster than N round-trips for typical N=100k bars.
    first_open = bar_ts_ms[0]
    last_close = bar_ts_ms[-1] + bar_size_ms
    pattern = (L2_PARQUET_DIR / sym).as_posix() + '/yyyymm=*/l2_*.parquet'
    sql = (f"SELECT ts, p_bid, p_ask, v_bid, v_ask "
           f"FROM read_parquet('{pattern}', union_by_name=true) "
           f"WHERE ts >= {first_open} AND ts < {last_close} ORDER BY ts ASC")
    try:
        with _duckdb_con() as con:
            df = con.execute(sql).fetchdf()
    except Exception as e:
        logger.warning("l2_feature_loader: bar-aligned read failed: %s", e)
        return out
    if df.empty:
        return out
    # Per-snapshot derived columns.
    df = df.copy()
    denom = (df['v_bid'] + df['v_ask']).clip(lower=1e-12)
    df['micro_price'] = (df['p_ask'] * df['v_bid'] + df['p_bid'] * df['v_ask']) / denom
    df['mid'] = (df['p_bid'] + df['p_ask']) / 2.0
    df['spread_bps'] = ((df['p_ask'] - df['p_bid']) / df['mid'].clip(lower=1e-12)) * 1e4
    df['book_imb'] = (df['v_bid'] - df['v_ask']) / denom
    df['delta_v_bid'] = df['v_bid'].diff().fillna(0.0)
    df['delta_v_ask'] = df['v_ask'].diff().fillna(0.0)
    df['ofi_step'] = df['delta_v_bid'] - df['delta_v_ask']
    df['ofi_5'] = df['ofi_step'].rolling(window=5, min_periods=1).sum()
    # Bucket assignment: ((ts - first_open) // bar_size_ms).
    df['bar_idx'] = ((df['ts'] - first_open) // bar_size_ms).astype('int64')
    df = df[(df['bar_idx'] >= 0) & (df['bar_idx'] < n)]
    if df.empty:
        return out
    grp = df.groupby('bar_idx')
    agg = grp.agg(
        micro_price=('micro_price', 'last'),
        spread_bps=('spread_bps', 'last'),
        book_imbalance=('book_imb', 'mean'),
        ofi_5=('ofi_5', 'last'),
        spread_p95_window=('spread_bps', lambda s: float(s.quantile(0.95))
                            if len(s) >= 2 else float(s.iloc[-1])),
        snapshot_count=('ts', 'count'),
    ).reset_index()
    for _, row in agg.iterrows():
        i = int(row['bar_idx'])
        out.at[i, 'l2_micro_price']       = float(row['micro_price'])
        out.at[i, 'l2_spread_bps']        = float(row['spread_bps'])
        out.at[i, 'l2_book_imbalance']    = float(row['book_imbalance'])
        out.at[i, 'l2_ofi_5']             = float(row['ofi_5'])
        out.at[i, 'l2_spread_p95_window'] = float(row['spread_p95_window'])
        out.at[i, 'l2_snapshot_count']    = int(row['snapshot_count'])
    if closes_list:
        nonzero = out['close'] > 0
        out.loc[nonzero, 'l2_micro_vs_close'] = (
            (out.loc[nonzero, 'l2_micro_price'] - out.loc[nonzero, 'close'])
            / out.loc[nonzero, 'close']
        )
    return out


__all__ = [
    "L2_FEATURE_COLUMNS",
    "L2_PARQUET_DIR",
    "l2_partitions_exist",
    "load_raw_window",
    "load_bar_aligned",
]
