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
    def total_fees(self) -> float:
        return sum(t.fees_paid + t.funding_paid for t in self.trades)

    @property
    def gross_pnl(self) -> float:
        return self.total_pnl + self.total_fees

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
            "gross_pnl_usdt": round(self.gross_pnl, 2),
            "total_fees_usdt": round(self.total_fees, 2),
            "win_rate_pct": round(self.win_rate * 100, 1),
            "sharpe": round(self.sharpe(), 3),
            "sortino": round(self.sortino(), 3),
            "calmar": round(self.calmar(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "profit_factor": round(self.profit_factor(), 2),
        }


def _market_impact_slippage(size_usdt: float, depth_usdt: float) -> float:
    """Square-root market impact model (Kyle's lambda approximation).

    Phase 4 slippage: instead of a flat percentage, slippage scales with
    sqrt(order_size / book_depth), reflecting how a larger order eats deeper
    into the L2 book.  Capped at 0.5% per side to prevent unrealistic fills.

    Typical depth_usdt values:
      BTC/USDT futures top-5 levels: ~500_000 USDT
      SOL/USDT: ~100_000 USDT
      ADA/USDT: ~50_000 USDT  (default)
    """
    ratio = max(size_usdt, 1.0) / max(depth_usdt, 1.0)
    return min(0.005, 0.001 * np.sqrt(ratio))


class Backtester:
    """
    Vectorized backtester. Feeds OHLCV + funding data through a signal function
    and simulates position management with realistic costs.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        maker_fee: float = 0.0002,          # 0.02% Binance futures maker
        taker_fee: float = 0.0004,          # 0.04% Binance futures taker
        slippage_pct: float | None = None,  # legacy override; None → depth model
        book_depth_usdt: float = 50_000.0,  # Phase 4: avg L2 depth for slippage model
        fee_preset: str | None = None,      # override both fees from FEE_PRESETS
        position_size_pct: float = 0.10,    # 10% of capital per trade
        max_hold_bars: int = 48,            # max hours to hold a position
    ):
        self.initial_capital = initial_capital
        if fee_preset and fee_preset in FEE_PRESETS:
            self.maker_fee = FEE_PRESETS[fee_preset]["maker"]
            self.taker_fee = FEE_PRESETS[fee_preset]["taker"]
        else:
            self.maker_fee = maker_fee
            self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct        # None = use depth model
        self.book_depth_usdt = book_depth_usdt
        # backward-compat alias
        self.fee_rate = self.taker_fee
        self.position_size_pct = position_size_pct
        self.max_hold_bars = max_hold_bars

    def _slip(self, size_usdt: float) -> float:
        """Return slippage fraction for a given order size."""
        if self.slippage_pct is not None:
            return self.slippage_pct
        return _market_impact_slippage(size_usdt, self.book_depth_usdt)

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

        # --- LATENCY ---
        # Shift signal by 1 bar to simulate realistic execution delay
        df[signal_col] = df[signal_col].shift(1).fillna(0.0)

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
                # --- SLIPPAGE on EXIT ---
                slip = self._slip(position["size_usdt"])
                exit_price = price * (1 - slip) if position["direction"] == 1 else price * (1 + slip)
                fee = position["size_usdt"] * self.maker_fee  # exit via limit order
                trade = TradeRecord(
                    symbol=symbol,
                    entry_time=position["entry_time"],
                    exit_time=row["timestamp"] if hasattr(row["timestamp"], "year") else datetime.now(_tz.utc),
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
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
                # GARCH position sizing: halve size on vol spike if enabled
                if "garch_size_mult" in row.index:
                    size *= float(row.get("garch_size_mult", 1.0))
                # MTF SMA-200 filter: skip entries against macro trend if filter col present
                if "signal_mtf_filter" in row.index:
                    mtf = float(row.get("signal_mtf_filter", 0.0))
                    if (signal > 0 and mtf < 0) or (signal < 0 and mtf > 0):
                        equity_series.append(equity)
                        continue
                direction = 1 if signal > 0 else -1
                # --- SLIPPAGE on ENTRY ---
                slip = self._slip(size)
                entry_price = price * (1 + slip) if direction == 1 else price * (1 - slip)
                entry_fee = size * self.taker_fee  # entry via market order
                position = {
                    "direction": direction,
                    "entry_price": entry_price,
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
            slip = self._slip(position["size_usdt"])
            exit_price = last["close"] * (1 - slip) if position["direction"] == 1 else last["close"] * (1 + slip)
            fee = position["size_usdt"] * self.taker_fee  # forced market exit at end
            trade = TradeRecord(
                symbol=symbol,
                entry_time=position["entry_time"],
                exit_time=last["timestamp"] if hasattr(last["timestamp"], "year") else datetime.now(_tz.utc),
                direction=position["direction"],
                entry_price=position["entry_price"],
                exit_price=exit_price,
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

    def walk_forward(
        self,
        df: pd.DataFrame,
        signal_col: str,
        strategy_name: str,
        symbol: str,
        n_folds: int = 5,
        funding_col: str = "funding_rate",
    ) -> dict:
        """
        Walk-forward validation: splits df into n_folds sequential test windows,
        runs the backtest on each, returns fold-by-fold and aggregate metrics.
        """
        n = len(df)
        fold_size = max(48, n // (n_folds + 2))
        fold_sharpes, fold_pnls, fold_win_rates = [], [], []

        for i in range(n_folds):
            start = (i + 1) * fold_size
            end   = min(start + fold_size, n)
            if end - start < 20:
                break
            fold_df = df.iloc[start:end].copy().reset_index(drop=True)
            r = self.run(fold_df, signal_col, f"{strategy_name}_f{i+1}", symbol, funding_col)
            fold_sharpes.append(r.sharpe())
            fold_pnls.append(r.total_pnl)
            fold_win_rates.append(r.win_rate)

        if not fold_sharpes:
            return {}

        return {
            "strategy":        strategy_name,
            "symbol":          symbol,
            "n_folds":         len(fold_sharpes),
            "wf_mean_sharpe":  round(float(np.mean(fold_sharpes)), 3),
            "wf_std_sharpe":   round(float(np.std(fold_sharpes)),  3),
            "wf_mean_pnl":     round(float(np.mean(fold_pnls)),    2),
            "wf_consistency":  round(sum(1 for p in fold_pnls if p > 0) / len(fold_pnls), 2),
            "wf_decay":        round(float(fold_pnls[-1] - fold_pnls[0]), 2) if len(fold_pnls) >= 2 else 0,
            "fold_pnls":       [round(p, 2) for p in fold_pnls],
        }


def _batch_ml_predict(df: pd.DataFrame, model_filename: str) -> pd.Series:
    """
    Batch-predict a trained joblib sklearn model on every bar of df.
    Returns a signal Series: +1 (long), -1 (short), 0 (hold/uncertain).
    Uses the model's feature_names_in_ to pick the right columns.
    """
    import joblib
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        return pd.Series(0.0, index=df.index)
    try:
        model = joblib.load(model_path)
        
        def find_features(obj, depth=0):
            if depth > 5 or obj is None: return None
            if hasattr(obj, "feature_names_in_"): return list(obj.feature_names_in_)
            for attr in ["estimator", "base_estimator", "best_estimator_", "model", "_final_estimator", "step"]:
                if hasattr(obj, attr):
                    res = find_features(getattr(obj, attr), depth + 1)
                    if res: return res
            if hasattr(obj, "calibrated_classifiers_"):
                for clf in getattr(obj, "calibrated_classifiers_"):
                    res = find_features(clf, depth + 1)
                    if res: return res
            if hasattr(obj, "steps"):
                for name, step in getattr(obj, "steps"):
                    res = find_features(step, depth + 1)
                    if res: return res
            return None

        feature_cols = find_features(model)
        
        if not feature_cols:
            meta_path = model_path.replace(".joblib", "_meta.json")
            if os.path.exists(meta_path):
                import json
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "features" in meta: feature_cols = meta["features"]
                    elif "feature_names" in meta: feature_cols = meta["feature_names"]
                except Exception:
                    pass

        if not feature_cols:
            return pd.Series(0.0, index=df.index)
        feat_df = df.copy()
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X = feat_df[feature_cols].fillna(0.0).values
        proba = model.predict_proba(X)
        # class index: 1 = bullish for binary classifiers
        if proba.shape[1] >= 2:
            bull_p = proba[:, 1]
        else:
            bull_p = proba[:, 0]
        signal = np.where(bull_p > 0.52, 1.0, np.where(bull_p < 0.48, -1.0, 0.0))
        return pd.Series(signal, index=df.index)
    except Exception as e:
        logger.debug("Batch ML predict failed (%s): %s", model_filename, e)
        return pd.Series(0.0, index=df.index)


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

    # ── Volatility Breakout (TTM Squeeze) ─────────────────────────────────────
    # BB inside Keltner = squeeze; breakout on volume surge
    df["signal_vol_breakout"] = 0.0
    if "kc_pos" in df.columns and "bb_pb" in df.columns:
        squeeze = (df["bb_pb"] < 0.15) & (df["kc_pos"].between(0.1, 0.9))
        vol_mean = df["volume"].rolling(20, min_periods=5).mean()
        vol_surge = df["volume"] > vol_mean * 1.5
        squeeze_prev = squeeze.shift(1).fillna(False).astype(bool)
        df.loc[squeeze_prev & (df["close"] > df["close"].shift(1)) & vol_surge,
               "signal_vol_breakout"] = 1.0
        df.loc[squeeze_prev & (df["close"] < df["close"].shift(1)) & vol_surge,
               "signal_vol_breakout"] = -1.0

    # ── Regime classifier series ──────────────────────────────────────────────
    df["signal_regime"] = 1  # default TRENDING
    try:
        from src.analysis.regime_classifier import RegimeClassifier
        clf = RegimeClassifier()
        if clf.is_ready and "timestamp" in df.columns:
            regimes = clf.predict_series(df)
            df["signal_regime"] = pd.Series(regimes, index=df.index).fillna(1).astype(int)
    except Exception as e:
        logger.debug("Regime series prediction skipped: %s", e)

    # ── ML batch signals (Base, Trend, Futures) ───────────────────────────────
    # Pre-build richer feature set for models
    from src.analysis.feature_engineering import (
        add_roc, add_atr, add_adx, add_taker_and_trade_features, add_time_features
    )
    df_ml = df.copy()
    df_ml["return"]            = df_ml["close"].pct_change()
    df_ml["log_return"]        = np.log(df_ml["close"] / df_ml["close"].shift(1))
    df_ml["volatility"]        = df_ml["return"].rolling(20, min_periods=5).std()
    df_ml["dist_sma_7"]        = df_ml["close"] / df_ml["close"].rolling(7).mean() - 1
    df_ml["dist_sma_30"]       = df_ml["close"] / df_ml["close"].rolling(30).mean() - 1
    df_ml["volume_momentum"]   = df_ml["volume"] / df_ml["volume"].rolling(20).mean().clip(lower=1e-9) - 1
    df_ml["atr_pct"]           = df_ml.get("atr_14", pd.Series(0.0, index=df_ml.index)) / df_ml["close"].clip(lower=1e-9)
    for lag in range(1, 6):
        df_ml[f"return_lag{lag}"] = df_ml["return"].shift(lag)
        df_ml[f"log_return_lag{lag}"] = df_ml["log_return"].shift(lag)
    df_ml["stoch_k"] = 0.0  # placeholder if not available
    df_ml["news_sentiment"] = 0.0
    df_ml["trend_alignment"] = (df_ml["close"] > df_ml["close"].rolling(50).mean()).astype(float)
    df_ml["volume_surge"]    = (df_ml["volume"] > df_ml["volume"].rolling(20).mean() * 1.5).astype(float)
    if "timestamp" in df_ml.columns:
        df_ml = add_time_features(df_ml)

    df["signal_base_ml"]    = _batch_ml_predict(df_ml, "btc_rf_model.joblib")
    df["signal_trend_ml"]   = _batch_ml_predict(df_ml, "trend_model.joblib")
    df["signal_futures_ml"] = _batch_ml_predict(df_ml, "futures_short_model.joblib")

    # ── Scalping ML (1m model, approximated on 1h data) ───────────────────────
    # Feature set mirrors FEATURE_COLUMNS from train_scalping_model.py.
    # Running a 1m-trained model on 1h bars is an approximation — signals are
    # directionally meaningful but magnitude/frequency differs vs live 1m mode.
    df_sc = df_ml.copy()
    df_sc["signal_rsi"] = df["signal_rsi"]
    df_sc["signal_bb"]  = df["signal_bb"]
    try:
        from src.analysis.feature_engineering import add_rsi as _add_rsi_sc
        df_sc = _add_rsi_sc(df_sc, 7)   # adds rsi_7
    except Exception:
        df_sc["rsi_7"] = df_sc["close"].ewm(span=7, adjust=False).mean()
    try:
        from src.analysis.feature_engineering import add_roc as _add_roc_sc
        df_sc = _add_roc_sc(df_sc, [3, 5, 10])   # adds roc_3, roc_5, roc_10
    except Exception:
        for _p in [3, 5, 10]:
            df_sc[f"roc_{_p}"] = df_sc["close"].pct_change(_p)
    # macd_fast = raw MACD line (EMA12 - EMA26)
    df_sc["macd_fast"] = (df_sc["close"].ewm(span=12, adjust=False).mean()
                          - df_sc["close"].ewm(span=26, adjust=False).mean())
    # dist_to_micro_supp — microstructure feature, not available at 1h → 0
    df_sc["dist_to_micro_supp"] = 0.0
    df["signal_scalping"] = _batch_ml_predict(df_sc, "scalping_model.joblib")

    # ── Elliott Wave proxy (vectorized approximation) ─────────────────────────
    # Impulse (Wave 3/5): strong 5-bar momentum above SMA-50, confirmed by ML
    sma50 = df["close"].rolling(50, min_periods=10).mean()
    mom5  = df["close"] / df["close"].shift(5).clip(lower=1e-9) - 1
    ml_bull = df["signal_base_ml"] > 0
    ml_bear = df["signal_base_ml"] < 0
    df["signal_elliott_proxy"] = 0.0
    df.loc[(mom5 > 0.02) & (df["close"] > sma50) & ml_bull, "signal_elliott_proxy"] =  1.0
    df.loc[(mom5 < -0.02) & (df["close"] < sma50) & ml_bear, "signal_elliott_proxy"] = -1.0

    # ── GARCH position size multiplier ────────────────────────────────────────
    # Proxy: recent 5-bar realized vol vs rolling 60-bar mean vol.
    # When current vol > 1.8x average → spike → use 0.5x size (like live bot).
    ret = df["close"].pct_change()
    rv5  = ret.rolling(5,  min_periods=2).std()
    rv60 = ret.rolling(60, min_periods=10).std().clip(lower=1e-9)
    df["garch_size_mult"] = np.where(rv5 / rv60 > 1.8, 0.5, 1.0)

    # ── MTF SMA-200 filter signal (1 = above SMA200, -1 = below) ─────────────
    sma200 = df["close"].rolling(200, min_periods=50).mean()
    df["signal_mtf_filter"] = np.where(df["close"] > sma200, 1.0, -1.0)

    # ── Ichimoku Cloud ────────────────────────────────────────────────────────
    try:
        from src.analysis.feature_engineering import add_ichimoku
        df = add_ichimoku(df)
    except Exception as e:
        logger.debug("Ichimoku signal skipped: %s", e)
        df["signal_ichimoku"] = 0.0

    # ── SuperTrend ────────────────────────────────────────────────────────────
    try:
        from src.analysis.feature_engineering import add_supertrend
        df = add_supertrend(df, period=10, multiplier=3.0)
    except Exception as e:
        logger.debug("Supertrend signal skipped: %s", e)
        df["signal_supertrend"] = 0.0

    # ── MACD Centerline + Divergence ──────────────────────────────────────────
    try:
        from src.analysis.feature_engineering import add_macd_divergence
        df = add_macd_divergence(df)
    except Exception as e:
        logger.debug("MACD divergence signal skipped: %s", e)
        df["signal_macd_div"] = 0.0

    # ── OU signals: entry (RANGING) and filter (not-stretched gate) ──────────────
    df["signal_ou_entry"] = 0.0
    df["ou_filter"] = 0.0
    try:
        rolling_mu  = df["close"].rolling(200, min_periods=50).mean()
        rolling_std = df["close"].rolling(200, min_periods=50).std().clip(lower=1e-9)
        deviation   = (df["close"] - rolling_mu) / rolling_std
        is_ranging  = df["signal_regime"] == 0  # only activate in RANGING regime
        # Entry: fade moves >1.5σ in RANGING regime
        df.loc[is_ranging & (deviation < -1.5), "signal_ou_entry"] =  1.0
        df.loc[is_ranging & (deviation >  1.5), "signal_ou_entry"] = -1.0
        # Filter: pass Ensemble B signal only when price is NOT stretched (|dev| < 2σ)
        # This shows the backtest value of filtering out overextended entries.
        df["ou_filter"] = df["signal_ensemble_b"] * (deviation.abs() < 2.0).astype(float)
    except Exception as e:
        logger.debug("OU signals skipped: %s", e)

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
    timeframes: tuple[str, ...] = ("1h",),
) -> pd.DataFrame:
    """
    Entry point: loads all watchlist data, runs all strategies (Group A + B)
    at every timeframe in `timeframes`, saves A/B comparison report.

    timeframes — iterable of bar TFs to backtest. Defaults to ('1h',) for
    backwards compatibility. Pass ('5m','1h','4h','1d','1w') after PR 1's
    multi-TF resample lands to compare strategy stability across TFs.
    Each comparison row is tagged with its `timeframe` so the dashboard's
    Stability view can group / rank by it.
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
    last_df = None  # used for walk-forward sample at the end

    # ── Outer loop: per timeframe ─────────────────────────────────────────
    for tf in timeframes:
        logger.info("=== Backtesting timeframe: %s ===", tf)
        for sym in symbols:
            df = None
            for fname in [f"{sym}_{tf}.csv.gz", f"{sym}_spot_{tf}.csv.gz"]:
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
                logger.warning("Skipping %s @ %s — insufficient data.", sym, tf)
                continue

            df = merge_funding_into_ohlcv(df, sym.replace("_", "/"))

            try:
                df = _build_signals(df)
            except Exception as e:
                logger.error("Signal build failed for %s @ %s: %s", sym, tf, e)
                continue

            # ── Run all registry-enabled backtest strategies ───────────────
            from src.engine.strategy_registry import enabled_backtest_signal_cols, is_enabled_backtest

            _A_ORIG = {"RSI_MeanReversion", "MACD_Momentum", "BB_Reversion", "Ensemble_A"}
            _bt_scalping = Backtester(initial_capital=initial_capital,
                                      fee_preset="scalping", max_hold_bars=6)
            for reg_name, label, sig_col in enabled_backtest_signal_cols():
                if sig_col not in df.columns:
                    continue
                try:
                    group_prefix = "A_" if reg_name in _A_ORIG else "B_"
                    _bt_run = _bt_scalping if reg_name == "Scalping_ML" else bt
                    res = _bt_run.run(df, sig_col, f"{group_prefix}{label}", sym)
                    # Tag the result with the timeframe so downstream rows
                    # carry it through (Stability comparison view groups
                    # by this column).
                    setattr(res, "timeframe", tf)
                    (group_a_results if group_prefix == "A_" else group_b_results).append(res)
                except Exception as e:
                    logger.error("Backtest failed for %s/%s @ %s: %s",
                                 sym, reg_name, tf, e)

            # ── Meta-filtered variants ────────────────────────────────────
            if is_enabled_backtest("MetaLabeler_Filter"):
                for strat, sig_col in [
                    ("RSI_MetaFiltered",      "signal_rsi"),
                    ("MACD_MetaFiltered",     "signal_macd"),
                    ("Ensemble_MetaFiltered", "signal_ensemble_b"),
                    ("Base_ML_MetaFiltered",  "signal_base_ml"),
                ]:
                    if sig_col not in df.columns:
                        continue
                    try:
                        filtered_signal = _apply_meta_filter(df, sig_col)
                        tmp_col = f"_tmp_meta_{strat}"
                        df[tmp_col] = filtered_signal
                        res = bt.run(df, tmp_col, strat, sym)
                        setattr(res, "timeframe", tf)
                        group_b_results.append(res)
                    except Exception as e:
                        logger.error("Meta-filtered backtest failed for %s/%s @ %s: %s",
                                     sym, strat, tf, e)

            last_df = df  # remember for the walk-forward sample below

    all_results = group_a_results + group_b_results
    comparison = bt.compare_strategies(all_results)

    if not comparison.empty:
        comparison["group"] = comparison["strategy"].apply(
            lambda s: "A_Original" if s.startswith("A_") else "B_New"
        )
        # Carry the per-result tf into the comparison frame. compare_strategies
        # may have already extracted it as a column; if not, project from
        # all_results in result-order (same length).
        if "timeframe" not in comparison.columns:
            comparison["timeframe"] = [getattr(r, "timeframe", "1h")
                                       for r in all_results[:len(comparison)]]

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

    # ── Walk-forward analysis ─────────────────────────────────────────────
    # Run WF on the last loaded df (representative sample). Tag with the
    # final TF so multi-TF runs produce per-TF WF rows for Stability view.
    wf_rows: List[dict] = []
    try:
        if last_df is not None and len(last_df) >= 200:
            for reg_name, label, sig_col in enabled_backtest_signal_cols():
                if sig_col not in last_df.columns:
                    continue
                wf = bt.walk_forward(last_df, sig_col, reg_name,
                                     sym if sym else "BTC_USDT")
                if wf:
                    wf["timeframe"] = timeframes[-1]
                    wf_rows.append(wf)
        if wf_rows:
            wf_path = os.path.join(output_dir, "wf_results.json")
            import json as _wfj
            with open(wf_path, "w", encoding="utf-8") as f:
                _wfj.dump(wf_rows, f, indent=2)
            logger.info("Walk-forward results saved: %d strategies", len(wf_rows))
    except Exception as exc:
        logger.warning("Walk-forward analysis failed: %s", exc)

    logger.info("Backtest complete. Results saved to %s", output_dir)
    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_full_backtest()
    print(results.to_string(index=False))
