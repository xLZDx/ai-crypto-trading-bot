"""
Quantitative Backtesting Engine — Phase 5.
Simulates all strategies on historical data and computes:
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Max Drawdown, Profit Factor
  - Funding cost (critical for futures strategies)
  - Per-trade unit economics
Profit formula: (Price_out - Price_in) × Size - Fees - Σ(Funding × Size)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone as _tz
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TradeRecord:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    size_usdt: float
    fees_paid: float
    funding_paid: float

    @property
    def pnl(self) -> float:
        raw = (self.exit_price - self.entry_price) / self.entry_price * self.direction * self.size_usdt
        return raw - self.fees_paid - self.funding_paid

    @property
    def return_pct(self) -> float:
        return self.pnl / self.size_usdt if self.size_usdt > 0 else 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    def sharpe(self, risk_free: float = 0.0) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        rets = self.equity_curve.pct_change().dropna()
        excess = rets - risk_free / 8760  # hourly risk-free
        std = excess.std()
        return float(np.sqrt(8760) * excess.mean() / std) if std > 0 else 0.0

    def sortino(self, risk_free: float = 0.0) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        rets = self.equity_curve.pct_change().dropna()
        excess = rets - risk_free / 8760
        downside = excess[excess < 0].std()
        return float(np.sqrt(8760) * excess.mean() / downside) if downside > 0 else 0.0

    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        roll_max = self.equity_curve.cummax()
        dd = (self.equity_curve - roll_max) / roll_max
        return float(dd.min())

    def calmar(self) -> float:
        mdd = abs(self.max_drawdown())
        if mdd == 0 or self.equity_curve.empty:
            return 0.0
        annual_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (8760 / len(self.equity_curve)) - 1
        return float(annual_return / mdd)

    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "n_trades": self.n_trades,
            "total_pnl_usdt": round(self.total_pnl, 2),
            "win_rate_pct": round(self.win_rate * 100, 1),
            "sharpe": round(self.sharpe(), 3),
            "sortino": round(self.sortino(), 3),
            "calmar": round(self.calmar(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "profit_factor": round(self.profit_factor(), 2),
        }


class Backtester:
    """
    Vectorized backtester. Feeds OHLCV + funding data through a signal function
    and simulates position management with realistic costs.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0004,       # 0.04% taker fee (Binance futures)
        position_size_pct: float = 0.10, # 10% of capital per trade
        max_hold_bars: int = 48,         # max hours to hold a position
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.position_size_pct = position_size_pct
        self.max_hold_bars = max_hold_bars

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str,
        strategy_name: str,
        symbol: str,
        funding_col: str = "funding_rate",
    ) -> BacktestResult:
        """
        Runs backtest on a DataFrame that contains:
          - close, open, high, low (price)
          - <signal_col>: numeric signal, positive = long, negative = short, 0 = flat
          - <funding_col>: 8h funding rate (0 if not a futures asset)
        """
        result = BacktestResult(strategy_name=strategy_name, symbol=symbol)
        df = df.copy().reset_index(drop=True)

        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="1h")

        if funding_col not in df.columns:
            df[funding_col] = 0.0

        equity = self.initial_capital
        equity_series = []
        position: Optional[dict] = None  # {direction, entry_price, entry_idx, size_usdt, entry_time}

        for i, row in df.iterrows():
            price = row["close"]
            signal = row.get(signal_col, 0.0)
            funding = row.get(funding_col, 0.0)

            # Accumulate funding cost if in position
            if position is not None:
                position["funding_accrued"] += abs(funding) * position["size_usdt"]
                position["bars_held"] += 1

            # Close conditions
            should_close = False
            if position is not None:
                direction = position["direction"]
                # Flip signal or max hold
                if (direction == 1 and signal < -0.1) or (direction == -1 and signal > 0.1):
                    should_close = True
                if position["bars_held"] >= self.max_hold_bars:
                    should_close = True

            if should_close and position is not None:
                fee = position["size_usdt"] * self.fee_rate
                trade = TradeRecord(
                    symbol=symbol,
                    entry_time=position["entry_time"],
                    exit_time=row["timestamp"] if hasattr(row["timestamp"], "year") else datetime.now(_tz.utc),
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=price,
                    size_usdt=position["size_usdt"],
                    fees_paid=position["entry_fee"] + fee,
                    funding_paid=position["funding_accrued"],
                )
                equity += trade.pnl
                result.trades.append(trade)
                position = None

            # Open new position
            if position is None and abs(signal) > 0.1 and equity > 0:
                size = equity * self.position_size_pct
                direction = 1 if signal > 0 else -1
                entry_fee = size * self.fee_rate
                position = {
                    "direction": direction,
                    "entry_price": price,
                    "entry_idx": i,
                    "size_usdt": size,
                    "entry_fee": entry_fee,
                    "entry_time": row["timestamp"] if hasattr(row["timestamp"], "year") else datetime.now(_tz.utc),
                    "bars_held": 0,
                    "funding_accrued": 0.0,
                }

            equity_series.append(equity)

        # Close any open position at end
        if position is not None and len(df) > 0:
            last = df.iloc[-1]
            fee = position["size_usdt"] * self.fee_rate
            trade = TradeRecord(
                symbol=symbol,
                entry_time=position["entry_time"],
                exit_time=last["timestamp"] if hasattr(last["timestamp"], "year") else datetime.now(_tz.utc),
                direction=position["direction"],
                entry_price=position["entry_price"],
                exit_price=last["close"],
                size_usdt=position["size_usdt"],
                fees_paid=position["entry_fee"] + fee,
                funding_paid=position["funding_accrued"],
            )
            equity += trade.pnl
            result.trades.append(trade)
            if equity_series:
                equity_series[-1] = equity

        result.equity_curve = pd.Series(equity_series, index=df.index[:len(equity_series)])
        return result

    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Returns a ranked comparison table for all strategies."""
        rows = [r.summary() for r in results]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        return df


def run_full_backtest(
    raw_dir: str | None = None,
    output_dir: str | None = None,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Entry point: loads all watchlist data, runs each strategy, saves comparison report.
    Returns the comparison DataFrame.
    """
    import json as _json

    if raw_dir is None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "backtest")
    os.makedirs(output_dir, exist_ok=True)

    wl_path = os.path.join(PROJECT_ROOT, "data", "watchlist.json")
    if os.path.exists(wl_path):
        with open(wl_path, "r", encoding="utf-8") as f:
            symbols = [s.replace("/", "_") for s in _json.load(f)]
    else:
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

    from src.analysis.feature_engineering import add_rsi, add_macd, add_bollinger_bands, add_roc, add_atr
    from src.data_ingestion.funding_rate_downloader import merge_funding_into_ohlcv

    bt = Backtester(initial_capital=initial_capital)
    all_results: List[BacktestResult] = []

    for sym in symbols:
        df = None
        for fname in [f"{sym}_1h.csv.gz", f"{sym}_spot_1h.csv.gz"]:
            fpath = os.path.join(raw_dir, fname)
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    break
                except Exception as e:
                    logger.warning("Could not load %s: %s", fpath, e)

        if df is None or len(df) < 500:
            logger.warning("Skipping %s — insufficient data.", sym)
            continue

        df = merge_funding_into_ohlcv(df, sym.replace("_", "/"))

        # --- Feature engineering for signals ---
        df = add_rsi(df, 14)
        df = add_macd(df)
        df = add_bollinger_bands(df)
        df = add_roc(df, [7, 14])
        df = add_atr(df)

        # Strategy 1: RSI mean-reversion signal
        df["signal_rsi"] = 0.0
        df.loc[df["rsi_14"] < 30, "signal_rsi"] = 1.0
        df.loc[df["rsi_14"] > 70, "signal_rsi"] = -1.0

        # Strategy 2: MACD momentum
        df["signal_macd"] = 0.0
        df.loc[df["macd_hist"] > 0, "signal_macd"] = 1.0
        df.loc[df["macd_hist"] < 0, "signal_macd"] = -1.0

        # Strategy 3: Bollinger Band reversion
        df["signal_bb"] = 0.0
        df.loc[df["bb_pb"] < 0.1, "signal_bb"] = 1.0
        df.loc[df["bb_pb"] > 0.9, "signal_bb"] = -1.0

        # Strategy 4: Ensemble (average of all 3)
        df["signal_ensemble"] = (df["signal_rsi"] + df["signal_macd"] + df["signal_bb"]) / 3.0

        for strat, sig_col in [
            ("RSI_MeanReversion", "signal_rsi"),
            ("MACD_Momentum", "signal_macd"),
            ("BB_Reversion", "signal_bb"),
            ("Ensemble", "signal_ensemble"),
        ]:
            try:
                res = bt.run(df, sig_col, f"{strat}", sym)
                all_results.append(res)
            except Exception as e:
                logger.error("Backtest failed for %s/%s: %s", sym, strat, e)

    comparison = bt.compare_strategies(all_results)

    # Save results
    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M")
    comparison.to_csv(os.path.join(output_dir, f"comparison_{ts}.csv"), index=False)

    summary_path = os.path.join(output_dir, "latest_comparison.json")
    comparison.to_json(summary_path, orient="records", indent=2)
    logger.info("Backtest complete. Results saved to %s", output_dir)
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_full_backtest()
    print(results.to_string(index=False))
