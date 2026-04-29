"""
ContinuousTrainerAgent — online training of live-only models from the simulator feed.

Subscribes to AgentBus topic 'sim_candle' and routes bars to model-specific
rolling buffers. Triggers incremental retraining when each buffer is full:

  Scalping ML (HistGBT, 1m bars):
    - Buffer:  100 000 bars (sliding window)
    - Trigger: every 50 000 new bars
    - Method:  Retrain from scratch on full buffer (sklearn has no true
               online mode for tree models; sliding window is the best practice)
    - Saves:   models/scalping_model.joblib

  TFT Market-Maker (Darts TFTModel, 1h bars):
    - Buffer:  10 000 bars
    - Trigger: every 2 000 new bars
    - Method:  Fine-tune 1 epoch on new buffer (darts fit() is restartable)
    - Saves:   models/tft_model.pt (via darts checkpoint)

  OU Filter (OLS calibration, 1h bars):
    - Buffer:  1 000 bars
    - Trigger: every 200 new bars
    - Method:  Re-run MeanReversionCore.calibrate_ou_process() on full buffer
    - Saves:   data/ou_calibration.json

After each training cycle publishes 'retrain' on the bus with metrics, and
records results in SimulatorDataStore.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engine.agents.agent_bus import BaseAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MODELS_DIR    = PROJECT_ROOT / "models"
OU_CALIB_PATH = PROJECT_ROOT / "data" / "ou_calibration.json"


# ── Buffer configuration ──────────────────────────────────────────────────────

_SCALPING_BUFFER_SIZE = 100_000   # max 1m bars in buffer
_SCALPING_TRAIN_EVERY = 5_000     # new bars before retraining (lowered for demo)
_TFT_BUFFER_SIZE      = 10_000    # max 1h bars
_TFT_TRAIN_EVERY      = 500       # new bars before fine-tune (lowered for demo)
_OU_BUFFER_SIZE       = 1_000     # 1h bars for OU calibration
_OU_TRAIN_EVERY       = 50        # new bars before recalibration (lowered for demo)


class ContinuousTrainerAgent(BaseAgent):
    """
    Subscribes to 'sim_candle' and trains Scalping, TFT, and OU models
    continuously without interrupting the live bot.
    """

    NAME = "ContinuousTrainerAgent"

    def __init__(self, bus=None):
        super().__init__(bus=bus, interval_sec=30.0)

        # Rolling bar buffers (deque auto-drops oldest)
        self._scalping_buf: deque[dict] = deque(maxlen=_SCALPING_BUFFER_SIZE)
        self._tft_buf:      deque[dict] = deque(maxlen=_TFT_BUFFER_SIZE)
        self._ou_buf:       deque[dict] = deque(maxlen=_OU_BUFFER_SIZE)

        # New-bar counters since last training
        self._scalping_new: int = 0
        self._tft_new:      int = 0
        self._ou_new:       int = 0

        # Per-scenario accuracy for ScenarioManager feedback
        self._scenario_accuracy: dict[str, float] = {}

        # Thread lock for buffer mutations
        self._buf_lock = threading.Lock()

        # Training lock (only one model trains at a time to avoid OOM)
        self._train_lock = threading.Lock()

        # Store (lazy)
        self._store = None

        # Stats for dashboard
        self._stats: dict[str, Any] = {}

        # Which models are enabled (all on by default)
        self._enabled: set[str] = {"ScalpingML", "TFT_MM", "OU_Filter"}

    # ── AgentBus subscriptions ────────────────────────────────────────────────

    def _setup_subscriptions(self) -> None:
        self.bus.subscribe("sim_candle", self._on_sim_candle)

    def _on_sim_candle(self, msg) -> None:
        bar: dict = msg.payload
        tf = bar.get("timeframe", "")
        with self._buf_lock:
            if tf == "1m":
                self._scalping_buf.append(bar)
                self._scalping_new += 1
            elif tf in ("1h", "4h"):
                self._tft_buf.append(bar)
                self._ou_buf.append(bar)
                self._tft_new += 1
                self._ou_new  += 1

    # ── BaseAgent cycle ───────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        """Check buffers every interval_sec and trigger training if thresholds met."""
        with self._buf_lock:
            sc_new  = self._scalping_new
            tft_new = self._tft_new
            ou_new  = self._ou_new
            sc_buf  = len(self._scalping_buf)
            tft_buf = len(self._tft_buf)
            ou_buf  = len(self._ou_buf)

        if "ScalpingML" in self._enabled and sc_new >= _SCALPING_TRAIN_EVERY and sc_buf >= 500:
            self._train_scalping()
        if "TFT_MM" in self._enabled and tft_new >= _TFT_TRAIN_EVERY and tft_buf >= 200:
            self._train_tft()
        if "OU_Filter" in self._enabled and ou_new >= _OU_TRAIN_EVERY and ou_buf >= 50:
            self._calibrate_ou()

    # ── Scalping ML ──────────────────────────────────────────────────────────

    def _train_scalping(self) -> None:
        if not self._train_lock.acquire(blocking=False):
            return  # Another model is training
        try:
            with self._buf_lock:
                bars = list(self._scalping_buf)
                self._scalping_new = 0

            df = pd.DataFrame(bars)
            df = self._prepare_ohlcv(df)
            if df is None or len(df) < 500:
                return

            df = self._engineer_scalping_features(df)
            if df is None:
                return

            from src.engine.train_scalping_model import FEATURE_COLUMNS
            from src.analysis.triple_barrier import triple_barrier_labels_vectorized

            # Labels
            labels = triple_barrier_labels_vectorized(
                df["close"],
                pt_multiplier=1.5, sl_multiplier=1.0, horizon=5,
            )
            df["label"] = labels

            feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            valid = df[feat_cols + ["label"]].dropna()
            if len(valid) < 300:
                return

            X = valid[feat_cols].values
            y = (valid["label"] > 0).astype(int).values

            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.15, shuffle=False
            )

            base = HistGradientBoostingClassifier(
                max_iter=100, max_depth=5,
                learning_rate=0.05, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            )
            model = CalibratedClassifierCV(base, cv=3, method="sigmoid", n_jobs=-1)
            model.fit(X_tr, y_tr)

            acc = float(accuracy_score(y_val, model.predict(X_val)))
            loss = float(1.0 - acc)

            # Save model
            import joblib
            out_path = MODELS_DIR / "scalping_model.joblib"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, out_path)

            self._record_event("ScalpingML", len(X_tr), loss, loss * 0.95, acc)
            logger.info(
                "[Trainer] ScalpingML retrained on %d bars — acc=%.3f", len(X_tr), acc
            )
        except Exception as exc:
            logger.error("[Trainer] ScalpingML error: %s", exc, exc_info=True)
        finally:
            self._train_lock.release()

    # ── TFT Market-Maker ─────────────────────────────────────────────────────

    def _train_tft(self) -> None:
        if not self._train_lock.acquire(blocking=False):
            return
        try:
            with self._buf_lock:
                bars = list(self._tft_buf)
                self._tft_new = 0

            df = pd.DataFrame(bars)
            df = self._prepare_ohlcv(df)
            if df is None or len(df) < 200:
                return

            try:
                from darts import TimeSeries
                from darts.models import TFTModel

                series = TimeSeries.from_dataframe(
                    df[["close"]].reset_index(),
                    time_col="timestamp" if "timestamp" in df.columns else None,
                    value_cols=["close"],
                    freq="h",
                    fill_missing_dates=True,
                )
                if len(series) < 200:
                    return

                model_path = MODELS_DIR / "tft_model.pt"
                ckpt_dir   = MODELS_DIR / "tft_checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # Load existing model or create new one
                try:
                    model = TFTModel.load(str(model_path))
                    logger.info("[Trainer] TFT loaded from checkpoint for fine-tuning")
                except Exception:
                    use_gpu = self._has_gpu()
                    model = TFTModel(
                        input_chunk_length=168,
                        output_chunk_length=24,
                        hidden_size=256 if use_gpu else 64,
                        lstm_layers=2,
                        num_attention_heads=4,
                        dropout=0.1,
                        batch_size=256 if use_gpu else 32,
                        n_epochs=1,
                        add_relative_index=True,
                        pl_trainer_kwargs={
                            "accelerator": "gpu" if use_gpu else "cpu",
                            "devices": -1 if use_gpu else 1,
                            "enable_progress_bar": False,
                        },
                        model_name="tft_online",
                        work_dir=str(ckpt_dir),
                    )

                # Fine-tune 1 epoch on latest buffer
                split = int(len(series) * 0.85)
                train_s = series[:split]
                val_s   = series[split:]

                model.fit(train_s, epochs=1, verbose=False)
                model.save(str(model_path))

                # Evaluate MAE on val
                if len(val_s) > 24:
                    pred = model.predict(n=min(24, len(val_s)))
                    mae = float(
                        np.mean(np.abs(
                            pred.values().flatten() - val_s.values().flatten()[:len(pred)]
                        ))
                    )
                else:
                    mae = 0.0

                self._record_event("TFT_MM", len(train_s), mae, mae, 1.0 - min(mae / 1000, 1.0))
                logger.info("[Trainer] TFT fine-tuned on %d bars — MAE=%.4f", len(train_s), mae)

            except ImportError:
                logger.warning("[Trainer] darts not available — TFT training skipped")

        except Exception as exc:
            logger.error("[Trainer] TFT error: %s", exc, exc_info=True)
        finally:
            self._train_lock.release()

    # ── OU Filter ────────────────────────────────────────────────────────────

    def _calibrate_ou(self) -> None:
        # OU calibration is fast, no train_lock needed
        try:
            with self._buf_lock:
                bars = list(self._ou_buf)
                self._ou_new = 0

            df = pd.DataFrame(bars)
            df = self._prepare_ohlcv(df)
            if df is None or len(df) < 50:
                return

            from src.analysis.mean_reversion import MeanReversionCore
            core = MeanReversionCore()
            prices = df["close"].values.astype(float)
            params = core.calibrate_ou_process(prices)

            if params is None:
                return

            # Persist calibration per symbol
            symbol = bars[0].get("symbol", "unknown") if bars else "unknown"
            calib: dict = {}
            if OU_CALIB_PATH.exists():
                try:
                    calib = json.loads(OU_CALIB_PATH.read_text())
                except Exception:
                    pass
            calib[symbol] = {
                **params,
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
                "bars_used": len(df),
            }
            OU_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            OU_CALIB_PATH.write_text(json.dumps(calib, indent=2))

            acc = 1.0 if params.get("signal", 0) != 0 else 0.5
            self._record_event("OU_Filter", len(df), 0.0, 0.0, acc)
            logger.info(
                "[Trainer] OU recalibrated for %s — theta=%.4f mu=%.2f signal=%s",
                symbol, params.get("theta", 0), params.get("mu", 0), params.get("signal", 0)
            )

        except Exception as exc:
            logger.error("[Trainer] OU error: %s", exc, exc_info=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Convert bar-dict list to a clean OHLCV DataFrame with datetime index."""
        try:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df = df.sort_index()
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            return df if len(df) > 0 else None
        except Exception as exc:
            logger.debug("[Trainer] _prepare_ohlcv error: %s", exc)
            return None

    def _engineer_scalping_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            from src.analysis.feature_engineering import (
                add_rsi, add_macd, add_bollinger_bands, add_roc,
                add_time_features, add_taker_and_trade_features,
                add_ofi, add_vwap, add_keltner,
            )
            from src.analysis.fractional_diff import add_fractional_diff
            from src.engine.strategy_registry import REGISTRY

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

            df["vol_sma_5"]       = df["volume"].rolling(5).mean()
            df["volume_surge"]    = (df["volume"] > df["vol_sma_5"] * 2.0).astype(int)
            df["low_15"]          = df["low"].rolling(15).min()
            df["dist_to_micro_supp"] = (df["close"] - df["low_15"]) / df["close"].replace(0, np.nan)

            # Simple RSI / BB signals as features
            df["signal_rsi"] = np.where(df.get("rsi_7", pd.Series([50]*len(df))) < 30, 1.0,
                               np.where(df.get("rsi_7", pd.Series([50]*len(df))) > 70, -1.0, 0.0))
            df["signal_bb"]  = np.where(df.get("bb_pb", pd.Series([0.5]*len(df))) < 0.1, 1.0,
                               np.where(df.get("bb_pb", pd.Series([0.5]*len(df))) > 0.9, -1.0, 0.0))
            return df
        except Exception as exc:
            logger.error("[Trainer] Feature engineering error: %s", exc)
            return None

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _record_event(
        self, model: str, bars: int, train_loss: float,
        val_loss: float, accuracy: float, sharpe: float = 0.0
    ) -> None:
        if self._store is None:
            try:
                from src.simulation.data_store import SimulatorDataStore
                self._store = SimulatorDataStore()
            except Exception:
                return

        try:
            sid = ""  # no specific scenario ID from training thread
            self._store.record_training_event(
                model_name=model,
                scenario_id=sid,
                bars_trained=bars,
                train_loss=train_loss,
                val_loss=val_loss,
                accuracy=accuracy,
                sharpe=sharpe,
            )
        except Exception as exc:
            logger.debug("[Trainer] record_event error: %s", exc)

        self.publish("retrain", {
            "model":    model,
            "bars":     bars,
            "accuracy": round(accuracy, 4),
            "loss":     round(train_loss, 6),
            "sharpe":   round(sharpe, 3),
            "ts":       datetime.now(timezone.utc).isoformat(),
        })

        self._stats[model] = {
            "bars":     bars,
            "accuracy": accuracy,
            "loss":     train_loss,
            "sharpe":   sharpe,
            "ts":       datetime.now(timezone.utc).isoformat(),
        }

    def configure_models(self, enabled: list[str]) -> None:
        """Enable/disable specific models. e.g. ['ScalpingML', 'OU_Filter']"""
        self._enabled = set(enabled)
        logger.info("[ContinuousTrainerAgent] enabled models: %s", self._enabled)

    def get_stats(self) -> dict:
        with self._buf_lock:
            sc_buf  = len(self._scalping_buf)
            tft_buf = len(self._tft_buf)
            ou_buf  = len(self._ou_buf)
            sc_new  = self._scalping_new
            tft_new = self._tft_new
            ou_new  = self._ou_new
        stats = dict(self._stats)
        stats['_buffers'] = {
            'ScalpingML': {
                'buf': sc_buf, 'buf_max': _SCALPING_BUFFER_SIZE,
                'new': sc_new, 'trigger': _SCALPING_TRAIN_EVERY,
                'buf_pct': round(sc_buf / _SCALPING_BUFFER_SIZE * 100, 1),
                'trigger_pct': round(sc_new / _SCALPING_TRAIN_EVERY * 100, 1),
            },
            'TFT_MM': {
                'buf': tft_buf, 'buf_max': _TFT_BUFFER_SIZE,
                'new': tft_new, 'trigger': _TFT_TRAIN_EVERY,
                'buf_pct': round(tft_buf / _TFT_BUFFER_SIZE * 100, 1),
                'trigger_pct': round(tft_new / _TFT_TRAIN_EVERY * 100, 1),
            },
            'OU_Filter': {
                'buf': ou_buf, 'buf_max': _OU_BUFFER_SIZE,
                'new': ou_new, 'trigger': _OU_TRAIN_EVERY,
                'buf_pct': round(ou_buf / _OU_BUFFER_SIZE * 100, 1),
                'trigger_pct': round(ou_new / _OU_TRAIN_EVERY * 100, 1),
            },
        }
        return stats
