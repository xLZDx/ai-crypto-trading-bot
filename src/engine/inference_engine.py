import threading
import time
import logging
import pandas as pd
from pathlib import Path

from src.utils.model_integrity import verify_model_or_raise

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Dedicated background thread for Deep Learning inference (TFT + OFT).
    Runs models asynchronously so PyTorch operations don't block the HFT websocket loop.

    Phase 2 upgrade: also runs the Order Flow Transformer (OFT) when its
    checkpoint is available. OFT predictions are surfaced as
    `predictions[symbol]["oft"]` with keys: mu, sigma, p_move, liquidity_risk.
    The legacy TFT prediction stays at `predictions[symbol]["expected_return"]`
    for backwards compatibility with the existing dashboard.
    """
    def __init__(self, feature_store=None, model_path="models/tft_model.pt",
                 oft_model_path="models/oft_model.pt", update_interval=60):
        self.feature_store = feature_store
        self.model_path = Path(model_path)
        self.oft_model_path = Path(oft_model_path)
        self.update_interval = update_interval
        self.predictions = {}
        self._model = None
        self._oft_model = None
        self._oft_calibrator = None
        self._is_running = False
        self.lock = threading.Lock()

        # Pre-load both models (each is optional; missing model just disables that path)
        self._load_model()
        self._load_oft_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"TFT model not found at {self.model_path}. Inference disabled.")
            return

        try:
            verify_model_or_raise(str(self.model_path))
            from darts.models import TFTModel
            # Load PyTorch model (will map to GPU/CUDA if available)
            self._model = TFTModel.load(str(self.model_path))
            logger.info("TFT Model successfully loaded into Inference Engine.")
        except Exception as e:
            logger.error(f"Failed to load TFT Model: {e}")

    def _load_oft_model(self):
        """Load the Order Flow Transformer checkpoint (Phase 2)."""
        if not self.oft_model_path.exists():
            logger.info("OFT model not found at %s. OFT inference disabled.", self.oft_model_path)
            return
        try:
            import io
            import torch
            from src.utils.model_integrity import verify_and_load_bytes
            from src.models.order_flow_transformer import OrderFlowTransformer, OFTConfig
            # Phase A7 (2026-05-12): weights_only=True refuses to
            # unpickle arbitrary classes from the checkpoint —
            # closes the RCE-via-replaced-model-file vector flagged
            # by security review. Our checkpoint format here stores
            # only `config` (dict of basic types) + `state_dict`
            # (tensor dict), both of which weights_only allows.
            # Phase A8 (2026-05-12): verify_and_load_bytes returns
            # HMAC-verified bytes in a single open(), closing the
            # TOCTOU window vs. a separate verify_model_or_raise.
            buf = io.BytesIO(verify_and_load_bytes(str(self.oft_model_path)))
            ckpt = torch.load(buf, map_location="cpu", weights_only=True)
            cfg_dict = ckpt.get("config", {})
            cfg = OFTConfig(**{k: v for k, v in cfg_dict.items()
                              if k in OFTConfig.__dataclass_fields__})
            model = OrderFlowTransformer(cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._oft_model = model
            self._oft_calibrator = ckpt.get("calibrator")  # may be None
            logger.info("OFT model loaded into Inference Engine (cfg=%s).", cfg)
        except Exception as e:
            logger.warning("Failed to load OFT model: %s", e)
            self._oft_model = None

    def start(self, symbols: list):
        """Starts the background inference loop."""
        if not self._model:
            return
            
        self._is_running = True
        self.thread = threading.Thread(target=self._inference_loop, args=(symbols,), daemon=True)
        self.thread.start()
        logger.info("Inference Engine background thread started.")

    def _inference_loop(self, symbols: list):
        while self._is_running:
            for sym in symbols:
                try:
                    self._run_inference(sym)
                except Exception as e:
                    logger.error(f"Inference error on {sym}: {e}")
            time.sleep(self.update_interval)
            
    def _run_inference(self, symbol: str):
        """Run TFT (legacy) and OFT (Phase 2) on the latest features."""
        try:
            from darts import TimeSeries

            df = None
            if self.feature_store:
                df = self.feature_store.get_latest_data(symbol)
                if df is None or df.empty:
                    return

            # ── TFT path (legacy placeholder) ─────────────────────────────
            predicted_return = 0.005  # placeholder; existing dashboards expect this

            # ── OFT path (Phase 2) ────────────────────────────────────────
            oft_pred = self._oft_predict(df) if self._oft_model is not None else None

            with self.lock:
                self.predictions[symbol] = {
                    "expected_return": predicted_return,
                    "oft":             oft_pred,
                    "timestamp":       time.time(),
                }
        except Exception as e:
            logger.debug(f"Inference failed for {symbol}: {e}")

    def _oft_predict(self, df) -> dict | None:
        """Run a single OFT forward pass on the latest feature window.

        Returns: {mu, sigma, p_move, p_move_calibrated, liquidity_risk}
        """
        try:
            import numpy as np
            import torch
            cfg = self._oft_model.cfg
            # Build event tensor from kline-derived L2/microstructure features.
            # When real L2 columns are absent we fall back to zero-padding —
            # OFT then effectively conditions on cross-attention only.
            ev_cols = [c for c in df.columns
                       if c in {"ofi", "ob_ofi", "ob_imbalance", "ob_microprice",
                                "imbalance", "microprice", "volume", "ret_1",
                                "rsi_14", "macd", "vwap_dist", "funding_z",
                                "trades_count", "avg_trade_size",
                                "garch_volatility", "ou_mean_reversion"}]
            ob_cols = [c for c in df.columns
                       if c in {"ob_imbalance", "ob_microprice", "ob_ofi",
                                "imbalance", "microprice", "ofi"}]

            tail_e = df.tail(cfg.max_event_len).reindex(columns=ev_cols).fillna(0.0)
            tail_o = df.tail(cfg.max_orderbook_len).reindex(columns=ob_cols).fillna(0.0)

            ev = np.zeros((1, max(len(tail_e), 1), cfg.event_features), dtype=np.float32)
            ob = np.zeros((1, max(len(tail_o), 1), cfg.orderbook_features), dtype=np.float32)
            ev[0, :len(tail_e), :len(ev_cols)] = tail_e.to_numpy(dtype=np.float32)
            ob[0, :len(tail_o), :len(ob_cols)] = tail_o.to_numpy(dtype=np.float32)

            with torch.no_grad():
                out = self._oft_model(torch.from_numpy(ev), torch.from_numpy(ob))
            p_move = float(out.p_move.item())
            p_cal  = (float(self._oft_calibrator.transform([p_move])[0])
                      if self._oft_calibrator is not None else p_move)
            sigma  = float(torch.exp(0.5 * out.log_var).item())
            return {
                "mu":                float(out.mu.item()),
                "sigma":             sigma,
                "p_move":            p_move,
                "p_move_calibrated": p_cal,
                "liquidity_risk":    float(out.liquidity_risk.item()),
            }
        except Exception as exc:
            logger.debug("OFT predict failed: %s", exc)
            return None

    def get_latest_prediction(self, symbol: str):
        """Thread-safe way for main.py to fetch the latest neural net output."""
        with self.lock:
            return self.predictions.get(symbol, None)