#!/usr/bin/env python3
"""
Generate synthetic OHLCV data for Phase 6 smoke tests.

Writes to:
  data/parquet_test/base_test/   (~100 MB, AR(1) prices, >=50k rows)
  data/parquet_test/tft_test/    (~200 MB, GBM prices, >=200k rows)

Schema matches production: OHLCV_COLS = [timestamp, open, high, low, close,
  volume, quote_volume, trades_count, taker_buy_base, taker_buy_quote]
  with datetime64[us, UTC] timestamps and float64 OHLCV columns.

Partitioned: yyyymm=YYYY-MM/data_0.parquet per symbol per timeframe.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --rows-cpu 50000 --rows-gpu 200000
    python scripts/generate_synthetic_data.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_BASE    = PROJECT_ROOT / "data" / "parquet_test"

OHLCV_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "trades_count", "taker_buy_base", "taker_buy_quote",
]

SYMBOLS    = ["BTC_USDT", "ETH_USDT"]
TIMEFRAMES = ["1h", "4h"]

# AR(1) model parameters
AR_PHI   = 0.9999   # mean-reversion speed (close to 1 = slow revert = trending)
AR_SIGMA = 0.002    # per-bar log-return volatility

# GBM parameters for TFT test
GBM_MU    = 0.0002
GBM_SIGMA = 0.015


def _make_ohlcv_df(n_rows: int, start_price: float, freq: str,
                    model: str) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with n_rows rows."""
    start_ts = pd.Timestamp("2018-01-01", tz="UTC")
    timestamps = pd.date_range(start_ts, periods=n_rows, freq=freq, tz="UTC")

    rng = np.random.default_rng(seed=42)

    # Generate log returns
    if model == "ar1":
        log_ret = np.zeros(n_rows)
        log_mu  = np.log(start_price)
        log_p   = log_mu
        for i in range(n_rows):
            log_p        = AR_PHI * log_p + (1 - AR_PHI) * log_mu + rng.normal(0, AR_SIGMA)
            log_ret[i]   = log_p
        close = np.exp(log_ret)
    else:  # gbm
        z      = rng.standard_normal(n_rows)
        log_r  = (GBM_MU - 0.5 * GBM_SIGMA ** 2) + GBM_SIGMA * z
        close  = start_price * np.exp(np.cumsum(log_r))

    # Synthesize OHLCV from close
    noise    = rng.uniform(0.001, 0.003, n_rows)
    high     = close * (1 + noise)
    low      = close * (1 - noise)
    open_    = close * (1 + rng.uniform(-0.002, 0.002, n_rows))
    volume   = rng.lognormal(10, 1, n_rows)  # correlated loosely with volatility

    df = pd.DataFrame({
        "timestamp":      timestamps,
        "open":           open_.astype("float64"),
        "high":           high.astype("float64"),
        "low":            low.astype("float64"),
        "close":          close.astype("float64"),
        "volume":         volume.astype("float64"),
        "quote_volume":   (volume * close).astype("float64"),
        "trades_count":   rng.integers(100, 5000, n_rows).astype("float64"),
        "taker_buy_base": (volume * rng.uniform(0.3, 0.7, n_rows)).astype("float64"),
        "taker_buy_quote": (volume * close * rng.uniform(0.3, 0.7, n_rows)).astype("float64"),
    })

    # Assert dtypes before returning
    assert df["timestamp"].dtype == "datetime64[us, UTC]", f"bad timestamp dtype: {df['timestamp'].dtype}"
    for col in OHLCV_COLS[1:]:
        assert df[col].dtype == "float64", f"bad dtype for {col}: {df[col].dtype}"

    return df


def _write_partitioned(df: pd.DataFrame, dest_base: Path, dry_run: bool) -> None:
    """Write df partitioned by YYYY-MM to dest_base/yyyymm=YYYY-MM/data_0.parquet."""
    df = df.copy()
    df["_ym"] = df["timestamp"].dt.strftime("%Y-%m")

    schema = pa.Schema.from_pandas(df.drop(columns=["_ym"]), preserve_index=False)

    for ym, chunk in df.groupby("_ym"):
        chunk    = chunk.drop(columns=["_ym"]).reset_index(drop=True)
        part_dir = dest_base / f"yyyymm={ym}"
        out_file = part_dir / "data_0.parquet"

        if dry_run:
            log.info("  DRY-RUN: would write %d rows to %s", len(chunk), out_file)
            continue

        part_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        pq.write_table(table, str(out_file), compression="snappy")

    if not dry_run:
        log.info("  Written %d partitions to %s", df["_ym"].nunique(), dest_base)


def generate_set(name: str, n_rows: int, model: str,
                 dry_run: bool) -> None:
    """Generate one test dataset (e.g. 'base_test') for all symbols × timeframes."""
    dest_root = TEST_BASE / name
    log.info("Generating %s: %d rows/TF, model=%s -> %s", name, n_rows, model, dest_root)

    freq_map = {"1h": "1h", "4h": "4h"}
    start_prices = {"BTC_USDT": 30000.0, "ETH_USDT": 2000.0}

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            dest = dest_root / sym / tf
            log.info("  %s/%s", sym, tf)
            df = _make_ohlcv_df(n_rows, start_prices[sym], freq_map[tf], model)
            _write_partitioned(df, dest, dry_run)

    if not dry_run:
        total = sum(1 for _ in dest_root.rglob("*.parquet"))
        log.info("%s: %d parquet files written", name, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic parquet data for smoke tests")
    parser.add_argument("--rows-cpu", type=int, default=60_000, help="Rows per TF for CPU test")
    parser.add_argument("--rows-gpu", type=int, default=210_000, help="Rows per TF for GPU test (needs >=200k)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rows_cpu < 50_000:
        log.error("--rows-cpu must be >= 50000 (plan requirement)")
        sys.exit(1)
    if args.rows_gpu < 200_000:
        log.error("--rows-gpu must be >= 200000 (plan requirement: TFT needs ~1190 sequences)")
        sys.exit(1)

    generate_set("base_test", args.rows_cpu, "ar1",  dry_run=args.dry_run)
    generate_set("tft_test",  args.rows_gpu, "gbm",  dry_run=args.dry_run)

    log.info("Done. Run preflight_train.py to verify before training.")


if __name__ == "__main__":
    main()
