import threading
import time
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Dedicated background thread for Deep Learning inference (TFT/LSTM).
    Runs models asynchronously so PyTorch operations don't block the HFT websocket loop.
    """
    def __init__(self, feature_store=None, model_path="models/tft_model.pt", update_interval=60):
        self.feature_store = feature_store
        self.model_path = Path(model_path)
        self.update_interval = update_interval
        self.predictions = {}
        self._model = None
        self._is_running = False
        self.lock = threading.Lock()
        
        # Pre-load the model
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"TFT model not found at {self.model_path}. Inference disabled.")
            return
            
        try:
            from darts.models import TFTModel
            # Load PyTorch model (will map to GPU/CUDA if available)
            self._model = TFTModel.load(str(self.model_path))
            logger.info("✅ TFT Model successfully loaded into Inference Engine.")
        except Exception as e:
            logger.error(f"Failed to load TFT Model: {e}")

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
        """Simulates fetching the last 1000 candles and running Darts predict."""
        # Queries the DB/Feature Store for the latest normalized DataFrame
        try:
            from darts import TimeSeries
            
            if self.feature_store:
                df = self.feature_store.get_latest_data(symbol)
                if df.empty:
                    return
                # TODO: Convert df to TimeSeries and run self._model.predict()
                
            # For now, return a placeholder prediction
            predicted_return = 0.005 # e.g. +0.5% expected
            
            with self.lock:
                self.predictions[symbol] = {
                    "expected_return": predicted_return,
                    "timestamp": time.time()
                }
        except Exception as e:
            logger.debug(f"TFT Inference failed for {symbol}: {e}")

    def get_latest_prediction(self, symbol: str):
        """Thread-safe way for main.py to fetch the latest neural net output."""
        with self.lock:
            return self.predictions.get(symbol, None)