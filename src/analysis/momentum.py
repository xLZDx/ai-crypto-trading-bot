"""
Cross-Sectional Momentum Strategy.
Ranks all watchlist assets by N-day return, generates LONG signals for top performers
and SHORT signals for bottom performers. Market-neutral: profits from relative strength
regardless of overall market direction.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossSectionalMomentum:
    """
    Scans all assets in the universe every `rebalance_period` candles.
    Ranks by lookback-period return. Top `top_pct` → long, Bottom `bottom_pct` → short.
    Signal scale: +1.0 = strongest long, -1.0 = strongest short, 0 = hold.
    """

    def __init__(
        self,
        lookback: int = 20,
        top_pct: float = 0.30,
        bottom_pct: float = 0.30,
        rebalance_period: int = 24,
    ):
        self.lookback = lookback
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.rebalance_period = rebalance_period
        self._signals: Dict[str, float] = {}
        self._last_prices: Dict[str, float] = {}
        self._tick = 0

    def update(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Call every candle with a dict of {symbol: close_price}.
        Returns a dict of {symbol: signal} only on rebalance ticks; else returns cached signals.
        """
        for sym, px in prices.items():
            self._last_prices[sym] = px

        self._tick += 1
        if self._tick % self.rebalance_period != 0:
            return self._signals

        return self._rebalance(prices)

    def _rebalance(self, prices: Dict[str, float]) -> Dict[str, float]:
        if len(prices) < 3:
            return {}
        syms = list(prices.keys())
        px_arr = np.array([prices[s] for s in syms])
        # Signal = rank-normalized score in [-1, 1]
        ranks = pd.Series(px_arr).rank(pct=True)
        signals: Dict[str, float] = {}
        for sym, rank in zip(syms, ranks):
            if rank >= (1.0 - self.top_pct):
                signals[sym] = (rank - (1.0 - self.top_pct)) / self.top_pct
            elif rank <= self.bottom_pct:
                signals[sym] = -(self.bottom_pct - rank) / self.bottom_pct - 1e-9
            else:
                signals[sym] = 0.0
        logger.info("Momentum rebalance: %s", {s: f"{v:+.2f}" for s, v in signals.items()})
        self._signals = signals
        return signals

    def compute_from_history(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch computation on a wide DataFrame (columns = symbols, rows = timestamps).
        Returns a DataFrame of the same shape with momentum signals in [-1, 1].
        Used for backtesting.
        """
        returns = price_df.pct_change(self.lookback)
        signals = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
        for idx in range(len(returns)):
            row = returns.iloc[idx]
            valid = row.dropna()
            if len(valid) < 3:
                signals.iloc[idx] = 0.0
                continue
            ranks = valid.rank(pct=True)
            sig = pd.Series(0.0, index=price_df.columns)
            top_mask = ranks >= (1.0 - self.top_pct)
            bot_mask = ranks <= self.bottom_pct
            sig[top_mask] = (ranks[top_mask] - (1.0 - self.top_pct)) / self.top_pct
            sig[bot_mask] = -(self.bottom_pct - ranks[bot_mask]) / self.bottom_pct
            signals.iloc[idx] = sig
        return signals.fillna(0.0)

    def get_signal(self, symbol: str) -> float:
        return self._signals.get(symbol, 0.0)


def load_momentum_prices(raw_dir: str, symbols: List[str], timeframe: str = "1h", days: int = 60) -> Optional[pd.DataFrame]:
    """
    Loads close prices for all symbols from local CSV.gz files into a wide DataFrame.
    Used to bootstrap the momentum model from disk without the live feed.
    """
    from datetime import datetime, timedelta, timezone as _tz
    cutoff = datetime.now(_tz.utc).replace(tzinfo=None) - timedelta(days=days)
    frames: Dict[str, pd.Series] = {}

    for sym in symbols:
        for fname in [f"{sym}_{timeframe}.csv.gz", f"{sym}_spot_{timeframe}.csv.gz"]:
            fpath = os.path.join(raw_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath, usecols=["timestamp", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
                df = df[df["timestamp"] >= cutoff]
                if len(df) > 0:
                    frames[sym] = df.set_index("timestamp")["close"]
                    break
            except Exception as exc:
                logger.warning("Could not load %s: %s", fpath, exc)

    if not frames:
        return None

    wide = pd.DataFrame(frames)
    wide = wide.resample("1h").last().ffill()
    return wide
