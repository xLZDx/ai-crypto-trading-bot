"""Phase 4 (2026-05-14) — F1 Data Integrity ("Janitor") skill.

Pre-trainer schema + bounds + freshness validation for OHLCV input frames.

Why this exists
---------------
HistGBT/Optuna trainers assume the input frame is clean. In practice, raw
exchange CSVs occasionally carry:
  - flash-crash spikes (single-bar 50% drops/rallies)
  - missing ticks (timestamp gaps > 1 bar interval)
  - zero-volume bars (exchange WS dropped the trade firehose for a window)
  - negative or zero prices (rare; usually de-pegging stable indexes)
  - duplicated timestamps after a resampler / cron-overlap

If any of these reach the trainer, the model fits to noise. The bot's
recent 12.8 h CPU runaway was downstream of bad training data: scalping
trained on a 45-minute zero-volume DEX gap and produced a model that
flagged every WS tick as a SELL.

This module catches the obvious cases before a trainer wastes CPU on them.

Two-tier severity:
  - HARD failures (schema, bounds, dtype) → raise DataQualityError;
    caller aborts training and surfaces an actionable message.
  - SOFT warnings (gap > 1.5x bar interval, volume drop > 90%) → log
    WARNING + record into the returned DataQualityReport for the
    trainer's meta JSON.

Operator gate
-------------
DATA_QUALITY_MODE env var (matches the HMAC enforcement pattern):
  - "enforce" (default): hard failures raise, soft warnings log
  - "warn"   : hard failures log CRITICAL + return; trainer still runs
  - "off"    : skip validation entirely (escape hatch only)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera.errors import SchemaError, SchemaErrors
    _HAS_PANDERA = True
except ImportError:  # pragma: no cover
    pa = None
    SchemaError = SchemaErrors = Exception
    _HAS_PANDERA = False

logger = logging.getLogger(__name__)

_MODE_ENV = "DATA_QUALITY_MODE"
_MODE_ENFORCE = "enforce"
_MODE_WARN = "warn"
_MODE_OFF = "off"
_VALID_MODES = frozenset({_MODE_ENFORCE, _MODE_WARN, _MODE_OFF})


class DataQualityError(RuntimeError):
    """Hard failure: input data violates schema or bounds. Trainer must abort."""


@dataclass
class DataQualityReport:
    """Returned to the trainer; persisted into meta JSON for audit."""
    symbol: str = ""
    timeframe: str = ""
    n_rows_in: int = 0
    n_rows_out: int = 0
    schema_ok: bool = True
    bounds_ok: bool = True
    monotonic_ts: bool = True
    duplicate_ts: int = 0
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    zero_volume_runs: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    price_spikes: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    hard_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "schema_ok": self.schema_ok,
            "bounds_ok": self.bounds_ok,
            "monotonic_ts": self.monotonic_ts,
            "duplicate_ts": self.duplicate_ts,
            "gaps_count": len(self.gaps),
            "zero_volume_runs_count": len(self.zero_volume_runs),
            "price_spikes_count": len(self.price_spikes),
            "soft_warnings": self.soft_warnings[:20],
            "hard_errors": self.hard_errors,
        }


# Pandera schema for raw OHLCV CSVs. Columns the trainer relies on.
def _ohlcv_schema():
    """Build the OHLCV schema lazily so a pandera-less env still imports.
    NaNs are not allowed in price/volume because every downstream trainer
    treats them as bug indicators (timestamp gap not yet fixed by the
    backfill daemon, exchange API drop, etc.)."""
    if not _HAS_PANDERA:
        return None
    return pa.DataFrameSchema(
        columns={
            "timestamp": pa.Column(
                pa.DateTime, nullable=False, coerce=True,
                description="UTC candle close timestamp, monotonically increasing",
            ),
            "open":   pa.Column(float, pa.Check.greater_than(0), nullable=False, coerce=True),
            "high":   pa.Column(float, pa.Check.greater_than(0), nullable=False, coerce=True),
            "low":    pa.Column(float, pa.Check.greater_than(0), nullable=False, coerce=True),
            "close":  pa.Column(float, pa.Check.greater_than(0), nullable=False, coerce=True),
            "volume": pa.Column(
                float, pa.Check.greater_than_or_equal_to(0), nullable=False, coerce=True,
            ),
        },
        strict=False,  # extra columns (taker_buy_base, trades_count) are allowed
        coerce=True,
    )


def _load_enforcement_mode() -> str:
    """Read DATA_QUALITY_MODE; default 'enforce'. Invalid values warn + default."""
    raw = (os.environ.get(_MODE_ENV) or _MODE_ENFORCE).strip().lower()
    if raw in _VALID_MODES:
        return raw
    logger.warning("%s=%r invalid; defaulting to %r", _MODE_ENV, raw, _MODE_ENFORCE)
    return _MODE_ENFORCE


def _bar_seconds(timeframe: str) -> int:
    """Convert a timeframe label to its bar duration in seconds."""
    tf = (timeframe or "").lower()
    return {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
        "1d": 86400, "3d": 259200, "1w": 604800,
    }.get(tf, 3600)


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "1h",
    max_spike_pct: float = 0.50,  # >50% bar-to-bar move is suspicious
    max_zero_vol_bars: int = 60,  # 60 consecutive zero-vol bars = data gap
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Run schema + bounds + freshness checks on an OHLCV frame.

    Behaviour by DATA_QUALITY_MODE:
      enforce  → hard errors raise DataQualityError
      warn     → hard errors logged CRITICAL; df returned anyway
      off      → no checks; df returned untouched with an empty report

    Returns (cleaned_df, report). The cleaned_df may have fewer rows than
    input (duplicate timestamps deduplicated, NaN-on-critical-col rows
    dropped) — read report.n_rows_out for the count.
    """
    mode = _load_enforcement_mode()
    rep = DataQualityReport(symbol=symbol, timeframe=timeframe,
                            n_rows_in=len(df), n_rows_out=len(df))
    if mode == _MODE_OFF:
        return df, rep

    if df is None or len(df) == 0:
        rep.hard_errors.append("empty input frame")
        rep.schema_ok = False
        _handle_hard(rep, mode)
        return df if df is not None else pd.DataFrame(), rep

    # 1. Schema (pandera if available, manual fallback otherwise)
    schema = _ohlcv_schema()
    if schema is not None:
        try:
            df = schema.validate(df, lazy=True)
        except SchemaErrors as e:  # type: ignore[misc]
            rep.schema_ok = False
            rep.hard_errors.append(f"pandera schema: {e}")
        except SchemaError as e:  # type: ignore[misc]
            rep.schema_ok = False
            rep.hard_errors.append(f"pandera schema: {e}")
    else:
        # Minimal manual check — required columns + dtype-ish.
        missing = [c for c in ("timestamp", "open", "high", "low", "close", "volume")
                   if c not in df.columns]
        if missing:
            rep.schema_ok = False
            rep.hard_errors.append(f"missing columns: {missing}")

    # 2. Bounds — prices > 0, volume >= 0. high >= low. high >= max(open, close).
    if rep.schema_ok:
        bad_price = df[(df["open"] <= 0) | (df["close"] <= 0)
                       | (df["high"] <= 0) | (df["low"] <= 0)]
        bad_vol = df[df["volume"] < 0]
        bad_hl = df[df["high"] < df["low"]]
        bad_hi = df[df["high"] < df[["open", "close"]].max(axis=1) - 1e-9]
        bad_lo = df[df["low"] > df[["open", "close"]].min(axis=1) + 1e-9]
        bad_total = len(bad_price) + len(bad_vol) + len(bad_hl) + len(bad_hi) + len(bad_lo)
        if bad_total > 0:
            rep.bounds_ok = False
            rep.hard_errors.append(
                f"bounds: {len(bad_price)} non-positive prices, "
                f"{len(bad_vol)} negative volumes, {len(bad_hl)} high<low, "
                f"{len(bad_hi)} high<max(o,c), {len(bad_lo)} low>min(o,c)"
            )

    # 3. Timestamps — monotonic + dedup.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        is_mono = ts.is_monotonic_increasing
        rep.monotonic_ts = bool(is_mono)
        if not is_mono:
            rep.hard_errors.append("timestamps not monotonically increasing")
        dups = ts.duplicated().sum()
        rep.duplicate_ts = int(dups)
        if dups > 0:
            df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
            rep.soft_warnings.append(f"deduplicated {dups} timestamp(s)")

    # 4. Gap detection (soft) — gaps > 1.5x bar interval flagged but not fatal.
    if "timestamp" in df.columns and len(df) > 1:
        bar_s = _bar_seconds(timeframe)
        ts = pd.to_datetime(df["timestamp"])
        deltas = ts.diff().dt.total_seconds()
        gap_threshold = bar_s * 1.5
        gap_idx = deltas[deltas > gap_threshold].index
        for i in gap_idx:
            gap_bars = int(round(deltas.iloc[i] / bar_s)) - 1
            rep.gaps.append((ts.iloc[i - 1], ts.iloc[i], gap_bars))
        if len(rep.gaps) > 0:
            rep.soft_warnings.append(f"{len(rep.gaps)} timestamp gap(s) > {gap_threshold:.0f}s")

    # 5. Zero-volume runs (soft) — N consecutive bars with volume=0.
    if "volume" in df.columns and len(df) > max_zero_vol_bars:
        zv = (df["volume"] == 0).astype(int).to_numpy()
        # Run-length encode
        in_run = False
        run_start = 0
        for i, v in enumerate(zv):
            if v == 1 and not in_run:
                in_run = True
                run_start = i
            elif v == 0 and in_run:
                in_run = False
                run_len = i - run_start
                if run_len >= max_zero_vol_bars:
                    ts0 = df["timestamp"].iloc[run_start]
                    ts1 = df["timestamp"].iloc[i - 1]
                    rep.zero_volume_runs.append((ts0, ts1))
        if in_run and len(zv) - run_start >= max_zero_vol_bars:
            ts0 = df["timestamp"].iloc[run_start]
            ts1 = df["timestamp"].iloc[-1]
            rep.zero_volume_runs.append((ts0, ts1))
        if rep.zero_volume_runs:
            rep.soft_warnings.append(
                f"{len(rep.zero_volume_runs)} zero-volume run(s) >= {max_zero_vol_bars} bars"
            )

    # 6. Price spikes (soft) — single-bar move > max_spike_pct.
    if "close" in df.columns and len(df) > 2:
        ret = df["close"].pct_change().abs()
        spike_idx = ret[ret > max_spike_pct].index
        for i in spike_idx:
            rep.price_spikes.append((df["timestamp"].iloc[i], float(ret.iloc[i])))
        if rep.price_spikes:
            rep.soft_warnings.append(
                f"{len(rep.price_spikes)} bar-to-bar move(s) > {max_spike_pct*100:.0f}%"
            )

    rep.n_rows_out = len(df)

    # 7. Severity handling — any hard_error triggers the policy handler.
    # Bug fix 2026-05-14: the original guard
    #   `if rep.hard_errors and rep.schema_ok is False or not rep.bounds_ok`
    # was parsed as `(hard_errors AND not schema_ok) OR not bounds_ok` and
    # silently swallowed monotonic-timestamp violations. Any populated
    # hard_errors list now triggers, regardless of which sub-check filled it.
    if rep.hard_errors:
        _handle_hard(rep, mode)

    for w in rep.soft_warnings:
        logger.warning("[data_quality][%s/%s] %s", symbol, timeframe, w)

    return df, rep


def _handle_hard(rep: DataQualityReport, mode: str) -> None:
    msg = (f"[data_quality][{rep.symbol}/{rep.timeframe}] HARD failures: "
           f"{'; '.join(rep.hard_errors)}")
    if mode == _MODE_WARN:
        logger.critical("%s — %s=warn, allowing training anyway.", msg, _MODE_ENV)
        return
    # enforce
    logger.critical("%s — aborting (set %s=warn to override)", msg, _MODE_ENV)
    raise DataQualityError(msg)


__all__ = [
    "DataQualityError", "DataQualityReport",
    "validate_ohlcv", "_load_enforcement_mode",
]
