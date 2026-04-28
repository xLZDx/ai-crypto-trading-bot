import os
import sys
import logging
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.safe_json import read_json
from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features, add_adx
)

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, model_filename='btc_rf_model.joblib', model_type='base'):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_dir, 'models', model_filename)
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        self.accuracy = 0.0
        self.long_accuracy = 0.0
        self.short_accuracy = 0.0
        self.last_error = ""
        self._last_confidence = 0.5
        
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info(f"ML Model loaded: {self.model_path}")

                meta_filename = model_filename.replace('.joblib', '_meta.json')
                meta_path = os.path.join(base_dir, 'models', meta_filename)
                meta = read_json(meta_path, default={})
                self.accuracy = meta.get("accuracy", 0.0)
                self.long_accuracy = meta.get("long_accuracy", 0.0)
                self.short_accuracy = meta.get("short_accuracy", 0.0)
            except Exception as e:
                logger.error(f"Failed to load ML model from {self.model_path}: {e}")
                self.last_error = f"Model load failed: {e}"
        else:
            logger.warning(f"ML Model not found: {self.model_path}. Predictions disabled.")
            self.last_error = f"Model file not found: {self.model_path}"

    def predict(self, data):
        self.last_error = ""
        if not self.is_loaded or not self.model:
            self.last_error = "Model not loaded"
            return None
        if len(data) < 30:
            return None
            
        try:
            df = pd.DataFrame(data)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['return'] = df['close'].pct_change()
            df = add_time_features(df)
            df = add_taker_and_trade_features(df)

            if self.model_type == 'scalping':
                df = add_rsi(df, period=7, col_name='rsi_7')
                _tmp = add_macd(df.copy(), fast=5, slow=13, signal=3, prefix='')
                df['macd_fast'] = _tmp['macd']
                df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)
                df['low_15'] = df['low'].rolling(15).min()
                df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']
                df = add_roc(df, [3, 5, 10])
                df = add_bollinger_bands(df, window=10)
                features = ['return', 'rsi_7', 'macd_fast', 'volume_surge', 'dist_to_micro_supp',
                            'taker_buy_ratio', 'avg_trade_size', 'hour', 'roc_3', 'roc_5', 'roc_10', 'bb_pb']

            elif self.model_type == 'futures':
                df = add_rsi(df, 14)
                df = add_roc(df, [5])
                df['low_30'] = df['low'].rolling(30).min()
                df['dist_to_support'] = (df['close'] - df['low_30']) / df['close']
                df['vol_sma_7'] = df['volume'].rolling(window=7).mean()
                df['volume_drop'] = (df['volume'] < df['vol_sma_7'] * 0.7).astype(int)
                features = ['return', 'rsi_14', 'dist_to_support', 'volume_drop', 'hour', 'roc_5']

            elif self.model_type == 'trend':
                df = add_macd(df)
                df = add_adx(df, 14)
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['sma_200'] = df['close'].rolling(window=200).mean()
                df['trend_alignment'] = (df['sma_50'] > df['sma_200']).astype(int)
                df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
                df['volume_surge'] = (df['volume'] > df['vol_sma_20'] * 1.5).astype(int)
                features = ['return', 'macd', 'macd_signal', 'macd_hist',
                            'trend_alignment', 'volume_surge', 'atr_14', 'adx_14']

            else:  # base model
                df['sma_7'] = df['close'].rolling(window=7).mean()
                df['sma_30'] = df['close'].rolling(window=30).mean()
                df['volatility'] = df['return'].rolling(window=7).std()
                df['dist_sma_7'] = df['close'] / df['sma_7'] - 1
                df['dist_sma_30'] = df['close'] / df['sma_30'] - 1
                df = add_rsi(df, 14)
                df = add_macd(df)
                df = add_bollinger_bands(df, window=20)
                df = add_roc(df, [3, 7, 14])
                df['vol_sma_14'] = df['volume'].rolling(window=14).mean()
                df['volume_momentum'] = df['volume'] / df['vol_sma_14']
                df['high_14'] = df['high'].rolling(window=14).max()
                df['low_14'] = df['low'].rolling(window=14).min()
                hl_diff = (df['high_14'] - df['low_14']).replace(0, 0.0001)
                df['stoch_k'] = (df['close'] - df['low_14']) / hl_diff * 100
                df['return_lag1'] = df['return'].shift(1)
                df['return_lag2'] = df['return'].shift(2)
                df['return_lag3'] = df['return'].shift(3)
                df['return_lag5'] = df['return'].shift(5)
                df['atr_pct'] = (df['high'] - df['low']) / df['close']
                df['news_sentiment'] = 0.0
                features = ['return', 'volatility', 'dist_sma_7', 'dist_sma_30', 'rsi_14',
                            'macd', 'macd_hist', 'volume_momentum', 'stoch_k',
                            'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
                            'atr_pct', 'taker_buy_ratio', 'avg_trade_size', 'hour', 'day_of_week',
                            'roc_14', 'roc_3', 'roc_7', 'bb_pb', 'news_sentiment']
            
            # Validate all expected features are present
            missing = [f for f in features if f not in df.columns]
            if missing:
                self.last_error = f"Missing features: {missing}"
                logger.error(self.last_error)
                return None

            nan_counts = df[features].iloc[-1].isna().sum()
            if nan_counts > 0:
                logger.debug(f"[{self.model_type}] Filling {nan_counts} NaN(s) with 0 before prediction.")
            df[features] = df[features].fillna(0)

            last_row = df.iloc[-1:]
            X = last_row[features]

            # Use calibrated predict_proba when available (models now wrapped with
            # CalibratedClassifierCV — returns well-calibrated P(win) in [0, 1])
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                p_long = float(proba[1]) if len(proba) > 1 else float(proba[0])
                self._last_confidence = p_long
                # Only act when confidence exceeds threshold — avoids noisy 50/50 calls
                if p_long >= 0.60:
                    return 1
                elif (1.0 - p_long) >= 0.60:
                    return 0
                else:
                    self.last_error = f"Low confidence ({p_long:.2f}) — no trade"
                    return None
            else:
                prediction = self.model.predict(X)
                result = int(prediction[0])
                self._last_confidence = 0.55
                if result not in (0, 1):
                    self.last_error = f"Unexpected prediction value: {result}"
                    logger.warning(self.last_error)
                    return None
                return result
        except Exception as e:
            error_msg = f"ML Prediction Error: {e}"
            logger.error(error_msg)
            self.last_error = error_msg
            return None

    def predict_proba_long(self, data) -> float:
        """
        Return P(long win) directly as a float in [0, 1].
        Used by KellySizer and RiskAgent for position sizing.
        Returns 0.5 (neutral) when model unavailable or features missing.
        """
        result = self.predict(data)
        return getattr(self, "_last_confidence", 0.5)