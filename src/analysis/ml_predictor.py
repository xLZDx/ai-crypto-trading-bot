import os
import logging
import joblib
import pandas as pd
import json

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, model_filename='btc_rf_model.joblib', model_type='base'):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_dir, 'models', model_filename)
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        self.accuracy = 0.0
        self.last_error = ""
        
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info(f"ML Model loaded successfully from {self.model_path}")
                
                # Load accuracy metadata
                meta_path = os.path.join(base_dir, 'models', 'model_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.accuracy = meta.get("accuracy", 0.0)
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        else:
            logger.warning(f"ML Model not found at {self.model_path}. ML predictions will be disabled.")

    def predict(self, data):
        # Require at least 30 candles to calculate the SMA_30 feature
        self.last_error = ""
        if not self.model or len(data) < 30:
            return None
            
        try:
            df = pd.DataFrame(data)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['return'] = df['close'].pct_change()
            
            # Common features
            if 'taker_buy_base' in df.columns:
                df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 0.0001)
            else:
                df['taker_buy_ratio'] = 0.5
                
            if 'trades_count' in df.columns:
                df['avg_trade_size'] = df['volume'] / df['trades_count'].replace(0, 1)
            else:
                df['avg_trade_size'] = 0.0
            
            if self.model_type == 'scalping':
                delta = df['close'].diff()
                df['rsi_7'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=6, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=6, adjust=False).mean())))
                
                exp1 = df['close'].ewm(span=5, adjust=False).mean()
                exp2 = df['close'].ewm(span=13, adjust=False).mean()
                df['macd_fast'] = exp1 - exp2
                
                df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
                df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)
                
                df['low_15'] = df['low'].rolling(15).min()
                df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']
                
                features = ['return', 'rsi_7', 'macd_fast', 'volume_surge', 'dist_to_micro_supp', 'taker_buy_ratio', 'avg_trade_size']
            else:
                df['sma_7'] = df['close'].rolling(window=7).mean()
                df['sma_30'] = df['close'].rolling(window=30).mean()
                df['volatility'] = df['return'].rolling(window=7).std()
                df['dist_sma_7'] = df['close'] / df['sma_7'] - 1
                df['dist_sma_30'] = df['close'] / df['sma_30'] - 1
                
                delta = df['close'].diff()
                df['rsi_14'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=13, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean())))
                
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
                df['vol_sma_14'] = df['volume'].rolling(window=14).mean()
                df['volume_momentum'] = df['volume'] / df['vol_sma_14']
                
                df['high_14'] = df['high'].rolling(window=14).max()
                df['low_14'] = df['low'].rolling(window=14).min()
                high_low_diff = (df['high_14'] - df['low_14']).replace(0, 0.0001)
                df['stoch_k'] = (df['close'] - df['low_14']) / high_low_diff * 100
                
                df['return_lag1'] = df['return'].shift(1)
                df['return_lag2'] = df['return'].shift(2)
                df['return_lag3'] = df['return'].shift(3)
                df['return_lag5'] = df['return'].shift(5)
                df['atr_pct'] = (df['high'] - df['low']) / df['close']
                
                features = ['return', 'volatility', 'dist_sma_7', 'dist_sma_30', 'rsi_14', 'macd', 'macd_hist', 'volume_momentum', 'stoch_k', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5', 'atr_pct', 'taker_buy_ratio', 'avg_trade_size']
            
            # Fill possible NaNs with zeros to guarantee a prediction
            df[features] = df[features].fillna(0)
            
            last_row = df.iloc[-1:]
            X = last_row[features]
            
            prediction = self.model.predict(X)
            return int(prediction[0])
        except Exception as e:
            error_msg = f"ML Prediction Error: {e}"
            logger.error(error_msg)
            self.last_error = error_msg
            return None