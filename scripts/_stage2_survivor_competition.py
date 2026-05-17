"""Stage 2: survivor competition.

After Stage 1 retrain completes, runs the backtester on every (model, tf)
cell with Binance futures fees + slippage, then ranks by NET Sharpe so the
top-3 most efficient strategies emerge for the 24h testnet live trading.

Usage:
  python scripts/_stage2_survivor_competition.py
  python scripts/_stage2_survivor_competition.py --fee-preset spot

Output:
  data/backtest/stage2_survivor_2026-05-16.csv
  data/backtest/stage2_top3.json    -- inputs for Stage 3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("d:/test 2/AI trading assistance")
sys.path.insert(0, str(ROOT))

from src.engine.backtester import run_full_backtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee-preset", default="futures",
                    help="Binance fee schedule (spot / spot_bnb / futures / futures_vip1 / scalping)")
    ap.add_argument("--initial-capital", type=float, default=10_000.0)
    ap.add_argument("--timeframes", nargs="+",
                    default=["5m", "15m", "1h", "4h", "1d"],
                    help="TFs to backtest (default: 5m 15m 1h 4h 1d)")
    args = ap.parse_args()

    print(f"=== Stage 2: survivor competition ===")
    print(f"  fee_preset={args.fee_preset}")
    print(f"  capital={args.initial_capital}")
    print(f"  timeframes={args.timeframes}")
    print()

    t0 = time.time()
    df = run_full_backtest(
        initial_capital=args.initial_capital,
        fee_preset=args.fee_preset,
        timeframes=tuple(args.timeframes),
        distribute=False,
    )
    elapsed = time.time() - t0
    print(f"\nbacktest finished in {elapsed/60:.1f} min")
    print(f"rows={len(df)} cols={list(df.columns)[:12]}")

    # Save raw output
    out_dir = ROOT / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / f"stage2_survivor_2026-05-16.csv"
    df.to_csv(raw_csv, index=False)
    print(f"raw CSV -> {raw_csv}")

    # Rank by NET Sharpe (the backtester's `sharpe` is already net of fees)
    # Tiebreaker: profit_factor, then -max_drawdown_pct.
    if "sharpe" not in df.columns:
        print("ERROR: 'sharpe' column missing -- backtester output schema changed")
        return 1

    # Filter: must have at least 30 trades to count
    qualified = df[df.get("n_trades", 0) >= 30].copy()
    if qualified.empty:
        print("WARN: no strategies with >=30 trades. Falling back to >=10.")
        qualified = df[df.get("n_trades", 0) >= 10].copy()
    if qualified.empty:
        print("ERROR: no strategies with enough trades to evaluate.")
        return 2

    qualified = qualified.sort_values(
        ["sharpe", "profit_factor"], ascending=[False, False]
    )

    print()
    print("=== TOP 10 by net Sharpe ===")
    cols = ["strategy", "symbol", "timeframe", "n_trades",
            "total_pnl_usdt", "sharpe", "sortino", "profit_factor",
            "max_drawdown_pct", "win_rate_pct"]
    show_cols = [c for c in cols if c in qualified.columns]
    print(qualified.head(10)[show_cols].to_string(index=False))
    print()

    top3 = qualified.head(3)
    print("=== TOP 3 (Stage 3 candidates) ===")
    top3_records = top3[show_cols].to_dict(orient="records")
    for i, rec in enumerate(top3_records, 1):
        print(f"  #{i}: {rec}")

    # Save top-3 picks for Stage 3
    out_json = out_dir / "stage2_top3.json"
    out_json.write_text(json.dumps({
        "ranked_at_unix": time.time(),
        "fee_preset": args.fee_preset,
        "min_trades_filter": int(qualified["n_trades"].min()),
        "top3": top3_records,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\ntop3 -> {out_json}")
    print("\nNext: python scripts/_stage3_testnet_live.py --top3-from", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
