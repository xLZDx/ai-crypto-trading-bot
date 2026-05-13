"""Replicate the bot's full scalping predict() call to find why it's 17 cols not 21."""
import sys, os, pandas as pd
sys.path.insert(0, r'D:/test 2/AI trading assistance')
os.environ.setdefault('MODEL_MANIFEST_KEY', '')

# Load real OHLCV for SOL/USDT (the bot was scalping it)
df = pd.read_csv(r'D:/test 2/AI trading assistance/data/raw/SOL_USDT_1m.csv.gz')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').tail(500).reset_index(drop=True)
print(f'data shape: {df.shape}, cols: {df.columns.tolist()}')

from src.analysis.ml_predictor import MLPredictor
mp = MLPredictor(model_filename='scalping_model.joblib', model_type='scalping')
print(f'is_loaded: {mp.is_loaded}')

# Call predict and capture errors
import logging
logging.basicConfig(level=logging.WARNING)
result = mp.predict(df)
print(f'\npredict() result: {result}')
print(f'last_status: {mp.last_status}')
print(f'last_error: {mp.last_error}')
