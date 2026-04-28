"""
Quantitative Backtesting Engine — Phase 6.
Simulates all strategies on historical data and computes:
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Max Drawdown, Profit Factor
  - Funding cost (critical for futures strategies)
  - Per-trade unit economics
Profit formula: (Price_out - Price_in) × Size - Fees - Σ(Funding × Size)

Fee model: entry uses taker_fee (market order), exit uses maker_fee (limit order).
Presets match actual Binance VIP-0 rates as of 2026.
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
_rng = np.random  # silence unused-import linters

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Binance fee presets (VIP-0, no BNB discount)
FEE_PRESETS = {
    "spot":         {"maker": 0.001,  "taker": 0.001},   # 0.10% both sides
    "spot_bnb":     {"maker": 0.00075,"taker": 0.00075}, # 0.075% with BNB
    "futures":      {"maker": 0.0002, "taker": 0.0004},  # 0.02% maker / 0.04% taker
    "futures_vip1": {"maker": 0.00016,"taker": 0.0004},  # VIP-1
    "futures_vip2": {"maker": 0.00014,"taker": 0.00035}, # VIP-2
    "scalping":     {"maker": 0.0002, "taker": 0.0004},  # same as futures (1m perp)
}


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
        maker_fee: float = 0.0002,       # 0.02% Binance futures maker
        taker_fee: float = 0.0004,       # 0.04% Binance futures taker
        fee_preset: str | None = None,   # override both fees from FEE_PRESETS
        position_size_pct: float = 0.10, # 10% of capital per trade
        max_hold_bars: int = 48,         # max hours to hold a position
    ):
        self.initial_capital = initial_capital
        if fee_preset and fee_preset in FEE_PRESETS:
            self.maker_fee = FEE_PRESETS[fee_preset]["maker"]
            self.taker_fee = FEE_PRESETS[fee_preset]["taker"]
        else:
            self.maker_fee = maker_fee
            self.taker_fee = taker_fee
        # backward-compat alias
        self.fee_rate = self.taker_fee
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
                fee = position["size_usdt"] * self.maker_fee  # exit via limit order
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
                entry_fee = size * self.taker_fee  # entry via market order
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
            fee = position["size_usdt"] * self.taker_fee  # forced market exit at end
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


def _build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all strategy signals on an OHLCV+funding DataFrame.
    Returns df with signal columns added.
    Group A = original 4 strategies (RSI, MACD, BB, Ensemble).
    Group B = new strategies + ML-filtered variants.
    """
    from src.analysis.feature_engineering import (
        add_rsi, add_macd, add_bollinger_bands, add_roc, add_atr,
        add_ofi, add_vwap, add_donchian, add_keltner, add_funding_zscore,
        add_liquidity_proximity, add_time_features
    )
    from src.analysis.fractional_diff import add_fractional_diff

    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_roc(df, [7, 14])
    df = add_atr(df)
    df = add_ofi(df)
    df = add_vwap(df)
    df = add_donchian(df, n=20)
    df = add_keltner(df)
    df = add_funding_zscore(df)
    df = add_liquidity_proximity(df)
    df = add_fractional_diff(df, d=0.4)
    if 'timestamp' in df.columns:
        df = add_time_features(df)

    # ── Group A: original strategies ──────────────────────────────────────
    df["signal_rsi"] = 0.0
    df.loc[df["rsi_14"] < 30, "signal_rsi"] = 1.0
    df.loc[df["rsi_14"] > 70, "signal_rsi"] = -1.0

    df["signal_macd"] = np.where(df["macd_hist"] > 0, 1.0, -1.0)

    df["signal_bb"] = 0.0
    df.loc[df["bb_pb"] < 0.1, "signal_bb"] = 1.0
    df.loc[df["bb_pb"] > 0.9, "signal_bb"] = -1.0

    df["signal_ensemble"] = (df["signal_rsi"] + df["signal_macd"] + df["signal_bb"]) / 3.0

    # ── Group B: new strategies ────────────────────────────────────────────
    # VWAP Reversion — long when price 0.5% below VWAP, short when 0.5% above
    df["signal_vwap"] = 0.0
    df.loc[df["vwap_dist"] < -0.005, "signal_vwap"] = 1.0
    df.loc[df["vwap_dist"] > 0.005, "signal_vwap"] = -1.0

    # Donchian breakout — long on 20-bar high break, short on low break
    df["signal_donchian"] = 0.0
    df.loc[df["don_pos_20"] > 0.98, "signal_donchian"] = 1.0
    df.loc[df["don_pos_20"] < 0.02, "signal_donchian"] = -1.0

    # Keltner breakout — momentum outside the channel
    df["signal_keltner"] = 0.0
    df.loc[df["kc_pos"] > 1.0, "signal_keltner"] = 1.0
    df.loc[df["kc_pos"] < 0.0, "signal_keltner"] = -1.0

    # Funding Arbitrage — short when funding > 0.1%, long when < -0.05%
    df["signal_funding"] = 0.0
    if "funding_rate" in df.columns:
        df.loc[df["funding_rate"] > 0.001, "signal_funding"] = -1.0   # shorts paid
        df.loc[df["funding_rate"] < -0.0005, "signal_funding"] = 1.0  # longs paid

    # Cross-sectional OFI momentum — strong buy/sell pressure via OFI
    df["signal_ofi"] = 0.0
    if "ofi_z" in df.columns:
        df.loc[df["ofi_z"] > 1.5, "signal_ofi"] = 1.0
        df.loc[df["ofi_z"] < -1.5, "signal_ofi"] = -1.0

    # Group B Ensemble
    b_signals = ["signal_vwap", "signal_donchian", "signal_keltner",
                 "signal_funding", "signal_ofi"]
    df["signal_ensemble_b"] = df[b_signals].mean(axis=1)

    return df


def _apply_meta_filter(df: pd.DataFrame, signal_col: str) -> pd.Series:
    """Apply meta-labeler filter to a signal column. Returns filtered signal."""
    try:
        from src.analysis.meta_labeler import MetaLabeler
        ml = MetaLabeler()
        if not ml.is_loaded:
            return df[signal_col]
        result = ml.batch_filter(df[signal_col], df)
        return pd.Series(result["filtered_signal"].values, index=df.index)
    except Exception as e:
        logger.debug("Meta-labeler filter skipped: %s", e)
        return df[signal_col]


def run_full_backtest(
    raw_dir: str | None = None,
    output_dir: str | None = None,
    initial_capital: float = 10_000.0,
    fee_preset: str = "futures",
) -> pd.DataFrame:
    """
    Entry point: loads all watchlist data, runs all strategies (Group A + B),
    saves A/B comparison report. Returns the combined comparison DataFrame.
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

    from src.data_ingestion.funding_rate_downloader import merge_funding_into_ohlcv

    bt = Backtester(initial_capital=initial_capital, fee_preset=fee_preset)
    group_a_results: List[BacktestResult] = []
    group_b_results: List[BacktestResult] = []

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

        try:
            df = _build_signals(df)
        except Exception as e:
            logger.error("Signal build failed for %s: %s", sym, e)
            continue

        # ── Group A: original strategies ──────────────────────────────────
        for strat, sig_col in [
            ("A_RSI_MeanReversion", "signal_rsi"),
            ("A_MACD_Momentum", "signal_macd"),
            ("A_BB_Reversion", "signal_bb"),
            ("A_Ensemble", "signal_ensemble"),
        ]:
            try:
                res = bt.run(df, sig_col, strat, sym)
                group_a_results.append(res)
            except Exception as e:
                logger.error("Backtest failed for %s/%s: %s", sym, strat, e)

        # ── Group B: new strategies ────────────────────────────────────────
        for strat, sig_col in [
            ("B_VWAP_Reversion", "signal_vwap"),
            ("B_Donchian_Breakout", "signal_donchian"),
            ("B_Keltner_Breakout", "signal_keltner"),
            ("B_Funding_Arb", "signal_funding"),
            ("B_OFI_Momentum", "signal_ofi"),
            ("B_Ensemble", "signal_ensemble_b"),
        ]:
            try:
                res = bt.run(df, sig_col, strat, sym)
                group_b_results.append(res)
            except Exception as e:
                logger.error("Backtest failed for %s/%s: %s", sym, strat, e)

        # ── Group B + Meta-filter ──────────────────────────────────────────
        for strat, sig_col in [
            ("B_RSI_MetaFiltered", "signal_rsi"),
            ("B_MACD_MetaFiltered", "signal_macd"),
            ("B_Ensemble_MetaFiltered", "signal_ensemble_b"),
        ]:
            try:
                filtered_signal = _apply_meta_filter(df, sig_col)
                df[f"_tmp_{strat}"] = filtered_signal
                res = bt.run(df, f"_tmp_{strat}", strat, sym)
                group_b_results.append(res)
            except Exception as e:
                logger.error("Meta-filtered backtest failed for %s/%s: %s", sym, strat, e)

    all_results = group_a_results + group_b_results
    comparison = bt.compare_strategies(all_results)

    if not comparison.empty:
        comparison["group"] = comparison["strategy"].apply(
            lambda s: "A_Original" if s.startswith("A_") else "B_New"
        )

    # ── A/B summary ───────────────────────────────────────────────────────
    if not comparison.empty and "group" in comparison.columns:
        for grp in ["A_Original", "B_New"]:
            sub = comparison[comparison["group"] == grp]
            if not sub.empty:
                logger.info("Group %s | mean Sharpe=%.3f | mean WinRate=%.1f%% | n_strategies=%d",
                            grp, sub["sharpe"].mean(), sub["win_rate_pct"].mean(), len(sub))

    ts = datetime.now(_tz.utc).strftime("%Y%m%d_%H%M")
    comparison.to_csv(os.path.join(output_dir, f"comparison_{ts}.csv"), index=False)
    comparison.to_json(os.path.join(output_dir, "latest_comparison.json"),
                       orient="records", indent=2)

    # Save A/B summary separately
    ab_path = os.path.join(output_dir, "ab_comparison.json")
    if not comparison.empty and "group" in comparison.columns:
        ab = comparison.groupby("group")[["sharpe", "sortino", "win_rate_pct",
                                         "max_drawdown_pct", "n_trades"]].mean()
        ab.to_json(ab_path, indent=2)

    logger.info("Backtest complete. Results saved to %s", output_dir)
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_full_backtest()
    print(results.to_string(index=False))
