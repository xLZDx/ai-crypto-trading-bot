import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_futures_data(filepath):
    print(f"Loading data for Futures (Shorting) Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # === Specific indicators for market crashes (Shorting) ===
    df['return'] = df['close'].pct_change()
    
    # RSI - look for hard overbought
    delta = df['close'].diff()
    df['rsi_14'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=13, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean())))
    
    # Distance to support (How high did we climb?)
    df['low_30'] = df['low'].rolling(30).min()
    df['dist_to_support'] = (df['close'] - df['low_30']) / df['close']
    
    # Volume drop momentum
    df['vol_sma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_drop'] = (df['volume'] < df['vol_sma_7'] * 0.7).astype(int)
    
    # TARGET: Will price drop by MORE than 1% in 3 candles? (Ideal for short)
    df['target_short'] = (df['close'].shift(-3) < df['close'] * 0.99).astype(int)
    
    df = df.dropna()
    return df

def train_futures_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        import json
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']
    
    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1h.csv.gz')
        if not os.path.exists(full_data_path):
            print(f"Warning: Data for {sym} not found at {full_data_path}. Auto-downloading...")
            import sys
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe='1h', days=6*365)

        if os.path.exists(full_data_path):
            df = prepare_futures_data(full_data_path)
            all_data.append(df)
            
    if not all_data:
        print("Error: No data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'rsi_14', 'dist_to_support', 'volume_drop']
    X = combined_df[feature_columns]
    y = combined_df['target_short']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training Futures Shorting AI Model...")
    # Train model to find rare but strong dumps
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200, max_depth=6, class_weight='balanced')
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"\nFutures Short Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'futures_short_model.joblib')
    joblib.dump(model, model_path)
    print(f"Futures Model saved to {model_path}")

if __name__ == "__main__":
    train_futures_model()