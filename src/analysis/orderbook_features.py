"""
L2/L3 order book features — Phase 1, Level 1 (Data Layer).

Formulas from updated_architecture_plan_en.md §2:

    # L2 Imbalance
    I = (V_bid - V_ask) / (V_bid + V_ask)

    # L2 Microprice
    P_micro = (P_ask * V_bid + P_bid * V_ask) / (V_bid + V_ask)

    # L3 / Flow Order Flow Imbalance
    OFI = delta_V_bid - delta_V_ask

All computations are causal: each row uses information up to and including
the current tick only. This module is the canonical L2/L3 feature path used
by Phase 2 OFT input pipelines and by the dashboard's Order Flow tab.

Distinct from `feature_engineering.add_ofi()` which uses kline-level
taker-buy proxies. Use this module when actual L2 snapshots are available.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Volume-floor to avoid division-by-zero on stale snapshots.
_EPS = 1e-12


def imbalance(v_bid, v_ask):
    """L2 order-book imbalance.  ∈ [-1, +1].

        I = (V_bid - V_ask) / (V_bid + V_ask)
    """
    v_bid = np.asarray(v_bid, dtype=float)
    v_ask = np.asarray(v_ask, dtype=float)
    denom = v_bid + v_ask
    safe = np.maximum(denom, _EPS)
    out = (v_bid - v_ask) / safe
    return np.where(denom > _EPS, out, 0.0)


def microprice(p_bid, p_ask, v_bid, v_ask):
    """Volume-weighted bid/ask midpoint (Stoll, Whaley).

        P_micro = (P_ask * V_bid + P_bid * V_ask) / (V_bid + V_ask)

    Falls back to the simple mid-price `(P_bid + P_ask)/2` when both
    volumes are zero (avoids NaNs on stale snapshots).
    """
    p_bid = np.asarray(p_bid, dtype=float)
    p_ask = np.asarray(p_ask, dtype=float)
    v_bid = np.asarray(v_bid, dtype=float)
    v_ask = np.asarray(v_ask, dtype=float)
    denom = v_bid + v_ask
    safe = np.maximum(denom, _EPS)
    micro = (p_ask * v_bid + p_bid * v_ask) / safe
    fallback = (p_bid + p_ask) / 2.0
    return np.where(denom > _EPS, micro, fallback)


def order_flow_imbalance(v_bid, v_ask) -> np.ndarray:
    """L3/flow OFI = ΔV_bid − ΔV_ask  (causal, no lookahead).

    The first row is 0 (no previous tick to difference against).
    """
    v_bid = np.asarray(v_bid, dtype=float)
    v_ask = np.asarray(v_ask, dtype=float)
    if v_bid.size == 0:
        return v_bid
    d_bid = np.diff(v_bid, prepend=v_bid[0])
    d_ask = np.diff(v_ask, prepend=v_ask[0])
    return d_bid - d_ask


def add_orderbook_features(
    df: pd.DataFrame,
    *,
    p_bid_col: str = "p_bid",
    p_ask_col: str = "p_ask",
    v_bid_col: str = "v_bid",
    v_ask_col: str = "v_ask",
    prefix: str = "",
) -> pd.DataFrame:
    """Add `imbalance`, `microprice`, `ofi` columns when bid/ask data is present.

    No-op (returns df unchanged) if any required column is missing — keeps
    candle-only pipelines working until the order-book collector is live.
    """
    needed = {v_bid_col, v_ask_col}
    if not needed.issubset(df.columns):
        return df

    p = lambda col: f"{prefix}{col}"
    df[p("imbalance")] = imbalance(df[v_bid_col], df[v_ask_col])
    df[p("ofi")] = order_flow_imbalance(df[v_bid_col], df[v_ask_col])

    if {p_bid_col, p_ask_col}.issubset(df.columns):
        df[p("microprice")] = microprice(
            df[p_bid_col], df[p_ask_col], df[v_bid_col], df[v_ask_col]
        )
    return df


def aggregate_levels(orderbook: dict, depth: int = 5) -> dict:
    """Reduce a full L2 snapshot {bids: [[p,v]], asks: [[p,v]], ...} to top-N.

    Returns: {symbol, ts, p_bid, p_ask, v_bid, v_ask, depth}
    """
    bids = orderbook.get("bids", [])[:depth]
    asks = orderbook.get("asks", [])[:depth]
    if not bids or not asks:
        return {}
    p_bid = float(bids[0][0])
    p_ask = float(asks[0][0])
    v_bid = float(sum(level[1] for level in bids))
    v_ask = float(sum(level[1] for level in asks))
    return {
        "symbol": orderbook.get("symbol", ""),
        "ts":     int(orderbook.get("timestamp") or 0),
        "p_bid":  p_bid,
        "p_ask":  p_ask,
        "v_bid":  v_bid,
        "v_ask":  v_ask,
        "depth":  depth,
    }


__all__ = [
    "imbalance",
    "microprice",
    "order_flow_imbalance",
    "add_orderbook_features",
    "aggregate_levels",
]
