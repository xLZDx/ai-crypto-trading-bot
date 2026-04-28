"""
QuantAgent — Quant / Math Reviewer.

Responsibilities:
  - Runs rolling backtest every N hours to detect strategy degradation
  - Compares live performance vs backtest expectations
  - Detects regime changes (correlation breakdown between symbols)
  - Flags overfitting: if live Sharpe < 50% of backtest Sharpe → alert
  - Publishes performance alerts to the bus
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

SHARPE_DEGRADATION_THRESHOLD = 0.5  # live Sharpe < 50% of backtest → alert
REVIEW_INTERVAL_HOURS = 6


class QuantAgent(BaseAgent):
    NAME = "QuantAgent"

    def __init__(self, bus=None, interval_sec: float = 3600.0 * REVIEW_INTERVAL_HOURS):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self._live_trades: list[dict] = []
        self._backtest_sharpe: dict[str, float] = {}
        self._last_review = datetime.now(timezone.utc)
        self._load_backtest_baseline()

    def _setup_subscriptions(self):
        self.bus.subscribe("order", self._on_order)

    def _on_order(self, msg) -> None:
        payload = msg.payload or {}
        if payload.get("status") == "closed":
            self._live_trades.append(payload)
            if len(self._live_trades) > 500:
                self._live_trades.pop(0)

    def _load_backtest_baseline(self) -> None:
        path = os.path.join(PROJECT_ROOT, "data", "backtest", "latest_comparison.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                records = json.load(f)
            for rec in records:
                key = f"{rec.get('strategy', '')}_{rec.get('symbol', '')}"
                self._backtest_sharpe[key] = float(rec.get("sharpe", 0))
            logger.info("[QuantAgent] Loaded backtest baseline: %d strategy-symbol pairs.",
                        len(self._backtest_sharpe))
        except Exception as e:
            logger.warning("[QuantAgent] Could not load backtest baseline: %s", e)

    def _compute_live_sharpe(self, trades: list[dict]) -> float:
        if len(trades) < 10:
            return 0.0
        pnls = [float(t.get("pnl", 0)) for t in trades if t.get("pnl") is not None]
        if not pnls:
            return 0.0
        arr = np.array(pnls)
        std = arr.std()
        return float(np.sqrt(8760) * arr.mean() / std) if std > 0 else 0.0

    def _run_cycle(self) -> None:
        live_sharpe = self._compute_live_sharpe(self._live_trades[-100:])

        alerts = []
        for key, bt_sharpe in self._backtest_sharpe.items():
            if bt_sharpe > 0 and live_sharpe < bt_sharpe * SHARPE_DEGRADATION_THRESHOLD:
                alerts.append({
                    "key": key, "bt_sharpe": bt_sharpe, "live_sharpe": live_sharpe
                })

        if alerts:
            logger.warning("[QuantAgent] Strategy degradation detected: %s", alerts)
            self.publish("perf", {
                "status": "degraded",
                "live_sharpe": live_sharpe,
                "alerts": alerts,
                "trigger_retrain": len(alerts) > 3,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            logger.info("[QuantAgent] Live Sharpe=%.3f — within acceptable range.", live_sharpe)
            self.publish("perf", {
                "status": "healthy",
                "live_sharpe": live_sharpe,
                "n_trades": len(self._live_trades),
            })

        # Check cross-asset correlation shift (BTC vs watchlist)
        self._check_correlation_shift()

    def _check_correlation_shift(self) -> None:
        """Detect correlation breakdown — signals regime instability."""
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        returns = {}
        for fname in os.listdir(raw_dir) if os.path.isdir(raw_dir) else []:
            if not fname.endswith("_1h.csv.gz"):
                continue
            sym = fname.replace("_1h.csv.gz", "")
            try:
                df = pd.read_csv(os.path.join(raw_dir, fname),
                                 usecols=["timestamp", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").tail(168)  # last 1 week
                returns[sym] = df["close"].pct_change().dropna().values
            except Exception:
                pass

        if len(returns) < 2 or "BTC_USDT" not in returns:
            return

        btc = returns["BTC_USDT"]
        low_corr = []
        for sym, ret in returns.items():
            if sym == "BTC_USDT" or len(ret) < 10:
                continue
            n = min(len(btc), len(ret))
            if n < 10:
                continue
            corr = float(np.corrcoef(btc[-n:], ret[-n:])[0, 1])
            if corr < 0.3:
                low_corr.append({"symbol": sym, "btc_corr": round(corr, 3)})

        if low_corr:
            logger.warning("[QuantAgent] Correlation breakdown detected: %s", low_corr)
            self.publish("perf", {
                "status": "correlation_breakdown",
                "low_corr_symbols": low_corr,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
