"""
Bulk parquet writer for market-context downloaders (OI, Fear&Greed, Liquidations).

Uses pd.to_parquet() directly into the DB partition structure — bypasses the
ParquetClient streaming buffer (which is for live tick-by-tick writes).
Partition layout mirrors _TABLES in parquet_client.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR       = PROJECT_ROOT / "data" / "db"


def _safe(v: str) -> str:
    return str(v).replace("/", "_").replace(" ", "_").replace(",", "_")


def write_context_parquet(
    df: pd.DataFrame,
    table: str,
    symbol: str | None = None,
) -> Path:
    """Write a DataFrame to the correct hot-DB partition for a context table.

    Tables: "open_interest", "fear_greed", "liquidations"

    - Splits by yyyymm and writes one file per month partition.
    - Merges with existing data (dedup on ts + symbol if applicable).
    - Returns the base table directory.
    """
    if df.empty:
        return DB_DIR / "hot" / table

    ts_col = "ts" if "ts" in df.columns else "timestamp"
    df = df.copy()
    if ts_col == "timestamp":
        df = df.rename(columns={"timestamp": "ts"})

    df["ts"] = pd.to_datetime(df["ts"])
    df["_yyyymm"] = df["ts"].dt.strftime("%Y%m")

    base = DB_DIR / "hot" / table
    base.mkdir(parents=True, exist_ok=True)

    dedup_cols = ["ts", "symbol"] if (symbol is not None and "symbol" in df.columns) else ["ts"]

    for yyyymm, chunk in df.groupby("_yyyymm"):
        chunk = chunk.drop(columns=["_yyyymm"]).reset_index(drop=True)

        if symbol is not None:
            sym_key = _safe(symbol)
            part_dir = base / f"symbol={sym_key}" / f"yyyymm={yyyymm}"
        else:
            part_dir = base / f"yyyymm={yyyymm}"

        part_dir.mkdir(parents=True, exist_ok=True)
        out_file = part_dir / "data.parquet"

        if out_file.exists():
            old = pd.read_parquet(out_file)
            old["ts"] = pd.to_datetime(old["ts"])
            chunk = pd.concat([old, chunk], ignore_index=True)
            chunk = chunk.sort_values("ts").drop_duplicates(subset=dedup_cols, keep="last")

        chunk.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")

    return base


def load_context_parquet(
    table: str,
    symbol: str | None = None,
    ts_col_out: str = "ts",
) -> pd.DataFrame:
    """Load all data for a context table (optionally filtered by symbol)."""
    base = DB_DIR / "hot" / table

    if symbol is not None:
        sym_key = _safe(symbol)
        glob = base / f"symbol={sym_key}" / "**" / "*.parquet"
    else:
        glob = base / "**" / "*.parquet"

    files = list(base.glob(
        f"symbol={_safe(symbol)}/**/*.parquet" if symbol else "**/*.parquet"
    ))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if ts_col_out != "ts":
        df = df.rename(columns={"ts": ts_col_out})
    return df
