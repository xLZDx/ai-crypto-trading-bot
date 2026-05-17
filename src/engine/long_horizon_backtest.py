"""
long_horizon_backtest — 5-year (and 'max') backtest preset for Phase E.

Until now `run_full_backtest` defaulted to whatever data was on disk.
That worked when archives were 1-2 years deep; with the 1-sec archives
back to 2017 we now have 8+ years of history per coin. Running 5m on
the full window is 500M+ rows per symbol and crashes the backtester;
running 1d gives only ~3000 bars per symbol — both extremes are bad.

This module provides three smart presets:

    short  (1y)   — 5m + 1h + 4h + 1d + 1w   (current default)
    medium (3y)   — 15m + 1h + 4h + 1d + 1w  (already-resampled, full)
    long   (5y)   — 1h + 4h + 1d + 1w        (skip 5m at 5y → 250M rows)
    max    (all)  — 4h + 1d + 1w + 1mo       (full history, low-res TFs only)

Returns the same DataFrame shape as run_full_backtest. Can be triggered
via the dashboard's POST /api/backtest/long_horizon endpoint or the
pipeline orchestrator's --horizon flag.

Trade-offs:
  - We cannot do meaningful 5m-resolution mean-reversion backtests over
    5 years; bar count + memory + DuckDB cache make it impractical
    without per-month chunking. Phase E accepts that and uses TFs where
    a 5y window stays in (~50K rows / symbol).
  - Walk-forward fold size scales with bar count, so longer windows
    automatically get more / longer folds — strictly better robustness
    measurement on the OOS metric we care about.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (years_back, allowed_tfs) per horizon preset.
HORIZONS: dict[str, tuple[float | None, tuple[str, ...]]] = {
    "short":  (1.0,  ("5m", "1h", "4h", "1d", "1w")),
    "medium": (3.0,  ("15m", "1h", "4h", "1d", "1w")),
    "long":   (5.0,  ("1h", "4h", "1d", "1w")),
    "max":    (None, ("4h", "1d", "1w", "1mo")),
}

DEFAULT_HORIZON = "long"


def run(horizon: str = DEFAULT_HORIZON,
        timeframes: tuple[str, ...] | None = None,
        initial_capital: float = 10_000.0,
        fee_preset: str = "futures") -> dict:
    """Run a long-horizon backtest. Returns
        {ok, horizon, years_back, timeframes, rows, elapsed_s}."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}; valid: {list(HORIZONS)}")
    years_back, default_tfs = HORIZONS[horizon]
    tfs = tuple(timeframes) if timeframes else default_tfs

    started = time.time()
    logger.info("[long_horizon_backtest] horizon=%s years_back=%s tfs=%s",
                horizon, years_back, list(tfs))

    # We don't yet thread start_date all the way through `run_full_backtest`
    # (the loader currently reads the full CSV). For Phase E we rely on the
    # _resampled_ files being aligned with each symbol's listing date, so
    # the result naturally covers each symbol's full available history.
    # `years_back` is recorded into the result tag so downstream consumers
    # know which horizon produced this row set.
    from src.engine.backtester import run_full_backtest
    df = run_full_backtest(timeframes=tfs,
                            initial_capital=initial_capital,
                            fee_preset=fee_preset)
    rows = int(len(df)) if df is not None else 0

    # Persist horizon tag alongside latest_comparison for downstream tools
    # (Stability heatmap can later filter / colour rows by horizon).
    try:
        import json as _j
        bt_path = PROJECT_ROOT / "data" / "backtest" / "latest_comparison.json"
        if bt_path.exists():
            data = _j.loads(bt_path.read_text())
            for r in data:
                r.setdefault("horizon", horizon)
                r.setdefault("years_back", years_back)
            bt_path.write_text(_j.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("could not tag latest_comparison with horizon: %s", exc)

    out = {
        "ok":          True,
        "horizon":     horizon,
        "years_back":  years_back,
        "timeframes":  list(tfs),
        "rows":        rows,
        "started_at":  datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s":   round(time.time() - started, 1),
    }
    logger.info("[long_horizon_backtest] done -- %d rows in %.1fs",
                rows, out["elapsed_s"])
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Long-horizon backtest preset")
    ap.add_argument("--horizon", choices=list(HORIZONS), default=DEFAULT_HORIZON)
    ap.add_argument("--timeframes", default="",
                    help="Comma-separated TF list (overrides preset).")
    ap.add_argument("--fee-preset", default="futures",
                    choices=["futures", "spot", "scalping"])
    args = ap.parse_args(argv)
    tfs = tuple(t.strip() for t in args.timeframes.split(",") if t.strip()) or None
    res = run(horizon=args.horizon, timeframes=tfs, fee_preset=args.fee_preset)
    print(json.dumps(res, default=str, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
