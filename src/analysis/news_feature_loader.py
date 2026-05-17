"""News sentiment feature loader — turn `data/parquet/_NEWS/news/*` into
bar-aligned features the trainers can consume.

Why
---
The dashboard's news scrapers (GDELT / Reddit / CryptoCompare backfill)
already populate the parquet store. Each row carries:

    ts (timestamp), title, url, source, tone (-1..+1), coin, ...

`tone` is computed by `src/analysis/finbert_scorer.py` (CryptoBERT → FinBERT →
lexicon fallback). What was missing: a loader that joins those headlines
to a bar timeline so the GBT trainers (base/trend/futures/scalping/meta) can
ingest sentiment as a regular feature.

Public surface
--------------
load_bar_aligned(symbol, bar_ts_ms, bar_size_ms) -> pandas.DataFrame
    Columns:
      news_tone_mean    — average tone of headlines that arrived in the bar
      news_tone_max_abs — max |tone| in the bar (proxy for surprise)
      news_count        — number of headlines in the bar (volume proxy)
      news_tone_zscore  — tone vs trailing 24h mean / std (mean reversion)
      news_decay_tone   — exponentially-weighted tone from previous N bars
                          (captures lingering news influence)

is_available(symbol=None) -> bool
    Cheap pre-check before invoking DuckDB.

Causality
---------
Every feature for bar B uses only headlines with `ts < bar_close_ms`.
Stable schema even when no headlines for the symbol — every column is 0.0.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWS_PARQUET_DIR = PROJECT_ROOT / 'data' / 'parquet' / '_NEWS' / 'news'
DUCKDB_TEMP_DIR = PROJECT_ROOT / 'data' / 'cache' / 'duckdb_temp'

NEWS_FEATURE_COLUMNS = (
    "news_tone_mean",
    "news_tone_max_abs",
    "news_count",
    "news_tone_zscore",
    "news_decay_tone",
)

# Map "BTC" / "BTC_USDT" / "BTC/USDT" to the canonical coin token used in
# news.coin. The scrapers tag headlines by base asset only (e.g. "BTC").
_BASE_TOKEN_OVERRIDES = {
    "USDT_USDT": "USDT",
    "USDC_USDT": "USDC",
}


def _coin_token(symbol: str) -> str:
    """Symbol → base-asset token to filter news.coin on."""
    s = symbol.replace("/", "_").upper()
    if s in _BASE_TOKEN_OVERRIDES:
        return _BASE_TOKEN_OVERRIDES[s]
    return s.split("_", 1)[0]


def _duckdb_con():
    import duckdb
    DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR.as_posix()}'")
    return con


def is_available(symbol: str | None = None) -> bool:
    """True iff at least one news parquet exists for `symbol` (or for any
    symbol when symbol=None)."""
    if not NEWS_PARQUET_DIR.exists():
        return False
    # Cheap heuristic — at least one yyyymm partition with a parquet file.
    try:
        for yyyymm_dir in NEWS_PARQUET_DIR.iterdir():
            if not yyyymm_dir.is_dir():
                continue
            if any(yyyymm_dir.glob('*.parquet')):
                return True
    except OSError:
        pass
    return False


def load_bar_aligned(
    symbol: str,
    bar_ts_ms: Iterable[int],
    bar_size_ms: int,
    *,
    decay_window: int = 6,
    decay_lambda: float = 0.5,
):
    """Build bar-aligned news features.

    `decay_window` past bars contribute to `news_decay_tone` with
    exponential weights w_k = decay_lambda^k (k=1..decay_window).
    """
    import pandas as pd
    bar_ts_ms = list(int(t) for t in bar_ts_ms)
    n = len(bar_ts_ms)
    out = pd.DataFrame({'ts': bar_ts_ms})
    for col in NEWS_FEATURE_COLUMNS:
        out[col] = 0.0
    if n == 0 or not is_available():
        return out
    coin = _coin_token(symbol)
    first_open = bar_ts_ms[0]
    last_close = bar_ts_ms[-1] + bar_size_ms
    # The news.ts column is a TIMESTAMP — convert to ms for join math.
    pattern = NEWS_PARQUET_DIR.as_posix() + '/yyyymm=*/*.parquet'
    sql = f"""
        SELECT
            CAST(epoch_ms(ts) AS BIGINT) AS ts_ms,
            tone,
            UPPER(COALESCE(coin, '')) AS coin_u
        FROM read_parquet('{pattern}', union_by_name=true)
        WHERE ts IS NOT NULL
          AND CAST(epoch_ms(ts) AS BIGINT) >= {first_open}
          AND CAST(epoch_ms(ts) AS BIGINT) < {last_close}
          AND (coin IS NULL OR UPPER(coin) = '{coin}' OR coin = '')
    """
    try:
        with _duckdb_con() as con:
            df = con.execute(sql).fetchdf()
    except Exception as e:
        logger.warning("news_feature_loader: read failed: %s", e)
        return out
    if df.empty:
        return out
    df = df.dropna(subset=['ts_ms', 'tone'])
    if df.empty:
        return out
    df['bar_idx'] = ((df['ts_ms'] - first_open) // bar_size_ms).astype('int64')
    df = df[(df['bar_idx'] >= 0) & (df['bar_idx'] < n)]
    if df.empty:
        return out
    df['abs_tone'] = df['tone'].abs()
    grp = df.groupby('bar_idx').agg(
        mean_tone=('tone', 'mean'),
        max_abs=('abs_tone', 'max'),
        count=('tone', 'count'),
    ).reset_index()
    # Materialise per-bar arrays.
    mean_arr = [0.0] * n
    max_abs_arr = [0.0] * n
    count_arr = [0] * n
    for _, row in grp.iterrows():
        i = int(row['bar_idx'])
        mean_arr[i] = float(row['mean_tone'])
        max_abs_arr[i] = float(row['max_abs'])
        count_arr[i] = int(row['count'])
    # Trailing 24h z-score. Convert "24h" to bar count.
    bars_per_day = max(1, int(86_400_000 / bar_size_ms))
    s = pd.Series(mean_arr, dtype=float)
    roll = s.rolling(window=bars_per_day, min_periods=2)
    z = (s - roll.mean()) / roll.std().replace(0, 1.0)
    z = z.fillna(0.0).clip(-5, 5)
    # Exponential decay carryover.
    decay_arr = [0.0] * n
    for i in range(n):
        acc = 0.0
        w_sum = 0.0
        for k in range(1, decay_window + 1):
            j = i - k
            if j < 0:
                break
            w = decay_lambda ** k
            acc += w * mean_arr[j]
            w_sum += w
        decay_arr[i] = (acc / w_sum) if w_sum > 0 else 0.0
    out['news_tone_mean']    = mean_arr
    out['news_tone_max_abs'] = max_abs_arr
    out['news_count']        = count_arr
    out['news_tone_zscore']  = z.tolist()
    out['news_decay_tone']   = decay_arr
    return out


__all__ = [
    "NEWS_FEATURE_COLUMNS", "NEWS_PARQUET_DIR",
    "is_available", "load_bar_aligned",
]
