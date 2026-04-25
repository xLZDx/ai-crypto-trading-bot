import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def prepare_scalping_data(filepath):
    print(f"Loading data for Scalping Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # === SHORT TERM SCALPING FEATURES (1m timeframe) ===
    df['return'] = df['close'].pct_change()
    
    # Fast RSI (7 minutes)
    delta = df['close'].diff()
    df['rsi_7'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=6, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=6, adjust=False).mean())))
    
    # Fast MACD (5, 13) for finding micro-bounces
    exp1 = df['close'].ewm(span=5, adjust=False).mean()
    exp2 = df['close'].ewm(span=13, adjust=False).mean()
    df['macd_fast'] = exp1 - exp2
    
    # Volume surges (Scalping on micro-pumps)
    df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)
    
    # Micro-support (Over 15 minutes)
    df['low_15'] = df['low'].rolling(15).min()
    df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']
    
    if 'taker_buy_base' in df.columns:
        df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 0.0001)
    else:
        df['taker_buy_ratio'] = 0.5
        
    if 'trades_count' in df.columns:
        df['avg_trade_size'] = df['volume'] / df['trades_count'].replace(0, 1)
    else:
        df['avg_trade_size'] = 0.0

    # TARGET: Will the price rise in the next 3 minutes?
    df['target_scalp'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    df = df.dropna()
    return df

def train_scalping_model():
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
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_1m.csv.gz')
        if not os.path.exists(full_data_path):
            print(f"Warning: 1m Data for {sym} not found at {full_data_path}. Auto-downloading...")
            import sys
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe='1m', days=70)

        if os.path.exists(full_data_path):
            df = prepare_scalping_data(full_data_path)
            # MEMORY LIMIT: Take only the last 100,000 candles (~70 days)
            # 6 years of 1m data will take tens of gigabytes of RAM and cause MemoryError!
            df = df.tail(100000)
            all_data.append(df)
            
    if not all_data:
        print("Error: No 1m data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'rsi_7', 'macd_fast', 'volume_surge', 'dist_to_micro_supp', 'taker_buy_ratio', 'avg_trade_size']
    X = combined_df[feature_columns]
    y = combined_df['target_scalp']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training Scalping AI Model...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"\nScalping Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'scalping_model.joblib')
    joblib.dump(model, model_path)
    print(f"Scalping Model saved to {model_path}")

if __name__ == "__main__":
    train_scalping_model()