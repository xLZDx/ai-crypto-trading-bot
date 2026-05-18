"""
Trading performance metrics — Sharpe, Profit Factor, Expectancy, MaxDD, win_rate.

Used by kpi_gate.py and champion_challenger.py to evaluate training runs.
All functions accept numpy arrays or pandas Series; return plain Python floats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    returns: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute trading performance metrics for a trained classifier.

    Parameters
    ----------
    y_true   : true labels (0/1)
    y_pred   : predicted labels (0/1)
    proba    : predicted probabilities for class 1 (optional; used for
               threshold-filtered metrics when provided)
    returns  : per-bar log-returns aligned with y_true
    threshold: confidence threshold; trades only where proba >= threshold
               (ignored when proba is None)

    Returns
    -------
    dict with keys: sharpe, profit_factor, expectancy, max_drawdown,
                    win_rate, n_trades, annualized_return
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    returns = np.asarray(returns, dtype=float)

    # Apply threshold filter if probabilities are available
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        active = proba >= threshold
    else:
        active = np.ones(len(y_pred), dtype=bool)

    # Trade returns: +return when prediction correct, -return when wrong
    # Long prediction (1): gain return if correct, lose return if wrong
    # Neutral (0 prediction): no trade
    trade_mask = active & (y_pred == 1)
    if trade_mask.sum() == 0:
        return _empty_metrics()

    trade_returns = np.where(y_true[trade_mask] == 1, returns[trade_mask], -returns[trade_mask])
    n_trades = int(trade_mask.sum())

    # Win rate
    wins = (trade_returns > 0).sum()
    win_rate = float(wins / n_trades) if n_trades > 0 else 0.0

    # Profit Factor: sum of gains / sum of losses
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float(gains * 10 if gains > 0 else 0.0)

    # Expectancy: mean trade return
    expectancy = float(trade_returns.mean())

    # Sharpe (annualised, assuming 252 trading days, 24h/day)
    mean_r = trade_returns.mean()
    std_r = trade_returns.std()
    if std_r > 0:
        # Scale by sqrt of trades-per-year; use 8760 for crypto (24/7)
        sharpe = float(mean_r / std_r * np.sqrt(8760))
    else:
        sharpe = 0.0

    # Maximum drawdown
    cum_ret = np.cumprod(1 + trade_returns)
    rolling_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Annualised return (simple)
    annualized_return = float(mean_r * 8760)

    return {
        "sharpe": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 6),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "annualized_return": round(annualized_return, 4),
    }


def _empty_metrics() -> dict:
    return {
        "sharpe": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "n_trades": 0,
        "annualized_return": 0.0,
    }
