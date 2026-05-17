"""Tick-level (1-second) feature loader.

The project's archives carry 1-second OHLCV bars in
`data/raw/historical/<SYM>_spot_1s.csv.gz` with columns:

    timestamp, open, high, low, close, volume, quote_volume,
    trades_count, taker_buy_base, taker_buy_quote

True print-level ticks are not on disk; 1s is the finest resolution we have.
This module treats 1s bars as the "tick" granularity and aggregates them
into bar-level microstructure intensity features for the target timeframe
(1m / 5m / 15m / 1h / 4h / 1d). Output columns:

    tick_count_mean         — avg trades_count per second inside the bar
    tick_count_p95          — 95th-pct of trades_count per second (burst proxy)
    taker_imbalance         — Σ taker_buy_base / Σ volume  (∈ [0, 1])
    taker_imbalance_drift   — (last_60s taker_imbalance) − (first_60s …)
    intra_bar_volatility    — Σ (high - low) / Σ close (per-second range / mid)
    volume_concentration    — top 10% of seconds' volume / total bar volume
    signed_volume_drift     — Σ sign(ret_1s) * volume_1s
    tick_seconds_count      — number of 1s rows that fell inside the bar
                              (sanity: small = sparse trading)

Stable-schema contract: when the 1s file is missing for the symbol, every
column is 0.0 and `tick_seconds_count = 0`. Trainers retain a constant
input shape.

Cache
-----
1s files are large (100+ MB compressed). For repeated calls during a sweep
we cache the parsed pandas DataFrame in-process keyed by (symbol, start_ms,
end_ms). Operator can flush with `clear_cache()`.
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIST_DIR = PROJECT_ROOT / 'data' / 'raw' / 'historical'
RAW_DIR  = PROJECT_ROOT / 'data' / 'raw'

TICK_FEATURE_COLUMNS = (
    "tick_count_mean",
    "tick_count_p95",
    "taker_imbalance",
    "taker_imbalance_drift",
    "intra_bar_volatility",
    "volume_concentration",
    "signed_volume_drift",
    "tick_seconds_count",
)

_cache: dict[tuple[str, int, int], object] = {}
_CACHE_MAX = 16


def clear_cache() -> None:
    _cache.clear()


PARQUET_DIR = PROJECT_ROOT / 'data' / 'parquet'


def _parquet_1s_dir(symbol: str) -> Path:
    sym = symbol.replace("/", "_").upper()
    return PARQUET_DIR / sym / "1s"


def _parquet_partitions_exist(symbol: str) -> bool:
    """True iff at least one yyyymm partition of 1s parquet exists for `symbol`.
    Set by scripts/migrate_1s_to_parquet.py."""
    d = _parquet_1s_dir(symbol)
    if not d.exists():
        return False
    try:
        for sub in d.iterdir():
            if sub.is_dir() and sub.name.startswith('yyyymm=') and any(sub.glob('*.parquet')):
                return True
    except OSError:
        pass
    return False


def _candidate_files(symbol: str) -> list[Path]:
    """Return existing 1s files for `symbol`, oldest priority first."""
    sym = symbol.replace("/", "_").upper()
    paths = [
        HIST_DIR / f"{sym}_spot_1s.csv.gz",
        HIST_DIR / f"{sym}_1s.csv.gz",
        RAW_DIR  / f"{sym}_1s.csv.gz",
    ]
    return [p for p in paths if p.exists()]


def has_tick_data(symbol: str) -> bool:
    """Either parquet 1s/ partitions OR a legacy gzip archive counts."""
    return _parquet_partitions_exist(symbol) or bool(_candidate_files(symbol))


def _read_1s_window(symbol: str, start_ms: int, end_ms: int):
    """Read 1s bars whose timestamp falls in [start_ms, end_ms).

    2026-05-15 — operator request: prefer the parquet 1s store
    (data/parquet/<SYM>/1s/yyyymm=*/data_*.parquet) over the gzip
    archives. Parquet supports cheap predicate pushdown so we read only
    the months we need; gzip requires full decompression. Falls back to
    the legacy gzip archive when parquet partitions are absent (i.e.
    before scripts/migrate_1s_to_parquet.py has been run).
    """
    import pandas as pd
    key = (symbol.upper(), int(start_ms), int(end_ms))
    if key in _cache:
        return _cache[key]

    # --- PARQUET FAST PATH ---
    if _parquet_partitions_exist(symbol):
        try:
            import duckdb
            DUCKDB_TEMP_DIR = PROJECT_ROOT / 'data' / 'cache' / 'duckdb_temp'
            DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect()
            con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR.as_posix()}'")
            pattern = (_parquet_1s_dir(symbol).as_posix() +
                       '/yyyymm=*/*.parquet')
            sql = f"""
                SELECT CAST(epoch_ms(timestamp) AS BIGINT) AS ts_ms,
                       high, low, close, volume,
                       COALESCE(trades_count, 0) AS trades_count,
                       COALESCE(taker_buy_base, 0.0) AS taker_buy_base
                FROM read_parquet('{pattern}', union_by_name=true)
                WHERE timestamp >= TIMESTAMP 'epoch' + INTERVAL ({int(start_ms)}) MILLISECOND
                  AND timestamp <  TIMESTAMP 'epoch' + INTERVAL ({int(end_ms)})   MILLISECOND
                ORDER BY ts_ms
            """
            df = con.execute(sql).fetchdf()
            con.close()
            if df is not None and not df.empty:
                if len(_cache) >= _CACHE_MAX:
                    _cache.pop(next(iter(_cache)))
                _cache[key] = df
                return df
        except Exception as exc:
            logger.warning("tick_feature_loader: parquet read failed (%s) -- falling through to gzip", exc)

    # --- GZIP LEGACY PATH ---
    files = _candidate_files(symbol)
    if not files:
        return None
    try:
        import duckdb
        DUCKDB_TEMP_DIR = PROJECT_ROOT / 'data' / 'cache' / 'duckdb_temp'
        DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR.as_posix()}'")
        path = files[0].as_posix()
        sql = f"""
            SELECT CAST(epoch_ms(timestamp) AS BIGINT) AS ts_ms,
                   high, low, close, volume,
                   COALESCE(trades_count, 0) AS trades_count,
                   COALESCE(taker_buy_base, 0.0) AS taker_buy_base
            FROM read_csv_auto('{path}', header=true)
            WHERE timestamp >= TIMESTAMP 'epoch' + INTERVAL ({int(start_ms)}) MILLISECOND
              AND timestamp <  TIMESTAMP 'epoch' + INTERVAL ({int(end_ms)})   MILLISECOND
            ORDER BY ts_ms
        """
        df = con.execute(sql).fetchdf()
        con.close()
    except Exception as exc:
        logger.warning("tick_feature_loader: duckdb read failed (%s) -- falling back to pandas streaming", exc)
        df = _pandas_stream_window(files[0], start_ms, end_ms)
    if df is None or df.empty:
        return df
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = df
    return df


def _pandas_stream_window(path: Path, start_ms: int, end_ms: int):
    """Fallback chunked CSV streaming when DuckDB isn't viable."""
    import pandas as pd
    rows = []
    try:
        for chunk in pd.read_csv(path, chunksize=200_000,
                                 parse_dates=['timestamp']):
            chunk['ts_ms'] = chunk['timestamp'].view('int64') // 1_000_000
            sub = chunk[(chunk['ts_ms'] >= start_ms) & (chunk['ts_ms'] < end_ms)]
            if not sub.empty:
                rows.append(sub[['ts_ms', 'high', 'low', 'close', 'volume',
                                 'trades_count', 'taker_buy_base']])
    except Exception as exc:
        logger.warning("tick_feature_loader: pandas stream failed: %s", exc)
        return None
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def load_bar_aligned(
    symbol: str,
    bar_ts_ms: Iterable[int],
    bar_size_ms: int,
):
    """Build bar-aligned tick (1s-derived) microstructure features."""
    import pandas as pd
    import numpy as np
    bar_ts_ms = list(int(t) for t in bar_ts_ms)
    n = len(bar_ts_ms)
    out = pd.DataFrame({'ts': bar_ts_ms})
    for c in TICK_FEATURE_COLUMNS:
        out[c] = 0.0
    if n == 0:
        return out
    df = _read_1s_window(symbol, bar_ts_ms[0], bar_ts_ms[-1] + bar_size_ms)
    if df is None or df.empty:
        return out
    df = df.copy()
    df['bar_idx'] = ((df['ts_ms'] - bar_ts_ms[0]) // bar_size_ms).astype('int64')
    df = df[(df['bar_idx'] >= 0) & (df['bar_idx'] < n)]
    if df.empty:
        return out
    df['vol_total'] = df['volume'].clip(lower=0)
    df['range_close'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-12)
    df['ret_1s'] = df.groupby('bar_idx')['close'].pct_change().fillna(0.0)
    df['signed_vol'] = np.sign(df['ret_1s']) * df['vol_total']

    grp = df.groupby('bar_idx')
    agg = grp.agg(
        n=('ts_ms', 'count'),
        tick_count_mean=('trades_count', 'mean'),
        tick_count_p95=('trades_count', lambda s: float(s.quantile(0.95)) if len(s) >= 2 else float(s.iloc[-1])),
        taker_sum=('taker_buy_base', 'sum'),
        vol_sum=('volume', 'sum'),
        range_sum=('range_close', 'sum'),
        signed_vol_sum=('signed_vol', 'sum'),
        first_half_taker=('taker_buy_base', lambda s: float(s.iloc[:max(1, len(s)//2)].sum())),
        last_half_taker=('taker_buy_base', lambda s: float(s.iloc[len(s)//2:].sum())),
        first_half_vol=('vol_total', lambda s: float(s.iloc[:max(1, len(s)//2)].sum())),
        last_half_vol=('vol_total', lambda s: float(s.iloc[len(s)//2:].sum())),
        vol_top_decile=('vol_total', lambda s: float(s.nlargest(max(1, len(s)//10)).sum())),
    ).reset_index()
    for _, row in agg.iterrows():
        i = int(row['bar_idx'])
        vol_sum = float(row['vol_sum']) or 1e-12
        out.at[i, 'tick_seconds_count']    = int(row['n'])
        out.at[i, 'tick_count_mean']       = float(row['tick_count_mean'])
        out.at[i, 'tick_count_p95']        = float(row['tick_count_p95'])
        out.at[i, 'taker_imbalance']       = float(row['taker_sum']) / vol_sum
        drift_first = float(row['first_half_taker']) / max(float(row['first_half_vol']), 1e-12)
        drift_last  = float(row['last_half_taker'])  / max(float(row['last_half_vol']),  1e-12)
        out.at[i, 'taker_imbalance_drift'] = drift_last - drift_first
        out.at[i, 'intra_bar_volatility']  = float(row['range_sum'])
        out.at[i, 'volume_concentration']  = float(row['vol_top_decile']) / vol_sum
        out.at[i, 'signed_volume_drift']   = float(row['signed_vol_sum'])
    return out


__all__ = [
    "TICK_FEATURE_COLUMNS",
    "has_tick_data", "load_bar_aligned", "clear_cache",
]
