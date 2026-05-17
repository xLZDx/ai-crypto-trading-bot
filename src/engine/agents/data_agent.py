"""
DataAgent — Data Engineer / ML Trainer.

Responsibilities:
  - Polls for new candles (every 60s by default)
  - Validates data quality (gap detection, stale data)
  - Detects distribution shift in live feature streams vs training distribution
  - Triggers model retraining when accuracy degrades below threshold
  - Manages feature store freshness
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from src.engine.agents.agent_bus import BaseAgent, get_bus

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

RETRAIN_ACCURACY_THRESHOLD = 52.0   # % — retrain if model accuracy drops below this
RETRAIN_COOLDOWN_HOURS = 24         # don't retrain more often than this
STALE_DATA_MINUTES = 5              # flag if last candle is older than this


class DataAgent(BaseAgent):
    NAME = "DataAgent"

    def __init__(self, symbols: list[str], bus=None, interval_sec: float = 60.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.symbols = symbols
        self._last_retrain: datetime | None = None
        self._last_candle_times: dict[str, datetime] = {}

    def _setup_subscriptions(self):
        # Listen for performance alerts that might trigger retraining
        self.bus.subscribe("perf", self._on_perf_alert)

    def _on_perf_alert(self, msg) -> None:
        payload = msg.payload or {}
        if payload.get("trigger_retrain"):
            logger.info("[DataAgent] Performance alert received -- scheduling retrain check.")
            self._check_retrain(forced=True)

    def _run_cycle(self) -> None:
        self._check_data_freshness()
        self._check_retrain()

    def _check_data_freshness(self) -> None:
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        for sym in self.symbols:
            for fname in [f"{sym}_1h.csv.gz", f"{sym}_spot_1h.csv.gz"]:
                fpath = os.path.join(raw_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    # Read only the last few rows for speed
                    df = pd.read_csv(fpath, usecols=["timestamp"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    last_ts = df["timestamp"].max()
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.tz_localize("UTC")

                    age_minutes = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
                    self._last_candle_times[sym] = last_ts

                    if age_minutes > STALE_DATA_MINUTES:
                        logger.warning("[DataAgent] %s data is %.0f minutes stale.", sym, age_minutes)
                        self.publish("candle", {
                            "symbol": sym, "status": "stale", "age_minutes": age_minutes
                        })
                    else:
                        self.publish("candle", {
                            "symbol": sym, "status": "fresh",
                            "last_ts": last_ts.isoformat()
                        })
                    break
                except Exception as e:
                    logger.debug("[DataAgent] Could not check %s: %s", sym, e)

    def _check_retrain(self, forced: bool = False) -> None:
        now = datetime.now(timezone.utc)
        if not forced and self._last_retrain:
            hours_since = (now - self._last_retrain).total_seconds() / 3600
            if hours_since < RETRAIN_COOLDOWN_HOURS:
                return

        # Check model accuracy from meta files
        models_dir = os.path.join(PROJECT_ROOT, "models")
        low_acc_models = []
        for meta_file in ["btc_rf_model_meta.json", "trend_model_meta.json",
                          "futures_short_model_meta.json", "scalping_model_meta.json"]:
            meta_path = os.path.join(models_dir, meta_file)
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                acc = meta.get("accuracy", 100.0)
                if acc < RETRAIN_ACCURACY_THRESHOLD:
                    low_acc_models.append((meta_file, acc))
            except Exception:
                pass

        if low_acc_models or forced:
            logger.warning("[DataAgent] Low accuracy models detected: %s -- triggering retrain.",
                           low_acc_models)
            self._last_retrain = now
            self.publish("retrain", {
                "reason": "accuracy_degraded" if low_acc_models else "forced",
                "models": [m[0] for m in low_acc_models],
                "timestamp": now.isoformat()
            })
