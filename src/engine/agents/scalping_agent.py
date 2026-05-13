"""
ScalpingAgent — 1-Minute Microstructure Specialist.

Timeframe: 1m
Models: Scalping model (HistGBT + Calibrated, Triple Barrier 5-bar)
Strategies: OFI momentum, VWAP micro-reversion, Keltner breakout
Risk: Strict slippage guard, fee-aware (must move > 0.08% to break even)
Fees: Binance futures taker 0.04% × 2 = 0.08% round-trip minimum

Key constraints:
  - Only trade BTC/ETH/SOL (deep enough liquidity to absorb 1m noise)
  - Minimum confidence 0.65 (higher bar vs spot/futures due to fee drag)
  - Max hold: 5 bars (5 minutes)
  - Block entry if spread > 0.05% of mid price
  - Never trade during volatile regime (slippage spikes)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

CONFIDENCE_THRESHOLD = 0.65       # higher bar for scalping
MAX_HOLD_BARS = 5                  # 5 minutes max hold
ROUND_TRIP_FEE = 0.0008           # 0.08% — min move to be profitable
LIQUID_SYMBOLS = {"BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"}


class ScalpingAgent(BaseAgent):
    NAME = "ScalpingAgent"

    def __init__(self, symbols: list[str], data_getter_1m, bus=None,
                 interval_sec: float = 60.0):
        super().__init__(bus=bus, interval_sec=interval_sec)
        self.symbols = [s for s in symbols if s in LIQUID_SYMBOLS]
        self.data_getter = data_getter_1m
        self._scalping_model = None
        self._load_model()
        if self.symbols != symbols:
            removed = set(symbols) - set(self.symbols)
            logger.info("[ScalpingAgent] Filtered illiquid symbols: %s", removed)

    def _load_model(self) -> None:
        try:
            import io
            import joblib
            from src.utils.model_integrity import verify_and_load_bytes
            path = os.path.join(PROJECT_ROOT, "models", "scalping_model.joblib")
            if os.path.exists(path):
                self._scalping_model = joblib.load(io.BytesIO(verify_and_load_bytes(path)))
                logger.info("[ScalpingAgent] Scalping model loaded.")
        except Exception as e:
            logger.warning("[ScalpingAgent] Model load error: %s", e)

    def _setup_subscriptions(self):
        self.bus.subscribe("regime", self._on_regime)

    def _on_regime(self, msg) -> None:
        regime = (msg.payload or {}).get("regime", 0)
        if regime == 2:
            logger.info("[ScalpingAgent] VOLATILE — scalping SUSPENDED (slippage risk).")

    def _get_regime(self) -> int:
        msg = self.bus.get_latest("regime")
        return (msg.payload or {}).get("regime", 0) if msg else 0

    def _run_cycle(self) -> None:
        regime = self._get_regime()
        if regime == 2:
            return  # No scalping in volatile regime

        for sym in self.symbols:
            try:
                df = self.data_getter(sym)
                if df is None or len(df) < 30:
                    continue

                signal, confidence = self._compute_scalping_signal(df, sym)
                if signal == 0 or confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Fee-adjusted break-even check
                last_price = float(df["close"].iloc[-1])
                atr = float(df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min())
                expected_move_pct = atr / last_price if last_price > 0 else 0
                if expected_move_pct < ROUND_TRIP_FEE * 1.5:
                    logger.debug("[ScalpingAgent] %s expected move %.4f%% < fee threshold — skip.",
                                 sym, expected_move_pct * 100)
                    continue

                logger.info("[ScalpingAgent] SCALP %s dir=%+d conf=%.2f move_est=%.3f%%",
                            sym, signal, confidence, expected_move_pct * 100)

                # 'trade_signal', not 'signal' — same recursion break as
                # SpotAgent/FuturesAgent. Scalping doesn't subscribe to 'signal'
                # so it wasn't part of the loop, but uses the same downstream
                # contract: RiskAgent only listens to 'trade_signal'.
                self.publish("trade_signal", {
                    "symbol": sym,
                    "direction": signal,
                    "confidence": confidence,
                    "strategy": "Scalping_OFI",
                    "market": "scalping",
                    "fee_preset": "scalping",
                    "meta_pass": True,
                    "max_hold_hours": MAX_HOLD_BARS / 60.0,
                    "size_mult": 0.5,   # smaller size for 1m scalps
                    "regime": regime,
                    "raw_signals": {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            except Exception as e:
                logger.error("[ScalpingAgent] Error on %s: %s", sym, e)

    def _compute_scalping_signal(self, df, sym: str) -> tuple[int, float]:
        """Use calibrated scalping model to generate signal + confidence."""
        if self._scalping_model is None:
            return 0, 0.0

        try:
            from src.analysis.feature_engineering import (
                add_rsi, add_macd, add_bollinger_bands, add_roc,
                add_time_features, add_taker_and_trade_features,
                add_ofi, add_vwap, add_keltner
            )
            from src.analysis.fractional_diff import add_fractional_diff

            df = df.copy()
            df["return"] = df["close"].pct_change()
            df = add_fractional_diff(df, d=0.4)
            df = add_rsi(df, period=7, col_name="rsi_7")
            df = add_macd(df, fast=5, slow=13, signal=3, prefix="")
            df.rename(columns={"macd": "macd_fast"}, errors="ignore", inplace=True)
            df = add_bollinger_bands(df, window=10)
            df = add_roc(df, [3, 5, 10])
            df = add_time_features(df)
            df = add_taker_and_trade_features(df)
            df = add_ofi(df, window=10)
            df = add_vwap(df)
            df = add_keltner(df, ema_period=10, atr_mult=1.5, atr_period=5)

            df["vol_sma_5"] = df["volume"].rolling(5).mean()
            df["volume_surge"] = (df["volume"] > df["vol_sma_5"] * 2.0).astype(int)
            df["low_15"] = df["low"].rolling(15).min()
            df["dist_to_micro_supp"] = (df["close"] - df["low_15"]) / df["close"]

            from src.engine.train_scalping_model import FEATURE_COLUMNS
            missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
            for col in missing:
                df[col] = 0.0

            last = df.iloc[[-1]][FEATURE_COLUMNS].fillna(0)
            if hasattr(self._scalping_model, "predict_proba"):
                proba = self._scalping_model.predict_proba(last)[0]
                p_long = float(proba[1]) if len(proba) > 1 else 0.5
                if p_long >= CONFIDENCE_THRESHOLD:
                    return 1, p_long
                elif (1 - p_long) >= CONFIDENCE_THRESHOLD:
                    return -1, 1 - p_long
            else:
                pred = int(self._scalping_model.predict(last)[0])
                return (1 if pred == 1 else -1), 0.55

        except Exception as e:
            logger.debug("[ScalpingAgent] Signal compute error for %s: %s", sym, e)

        return 0, 0.0
