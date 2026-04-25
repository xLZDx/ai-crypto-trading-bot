import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_trend_data(filepath):
    print(f"Loading data for Trend Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # === SPECIFIC TREND FOLLOWING FEATURES ===
    df['return'] = df['close'].pct_change()
    
    # 1. MACD (Trend Momentum - Key indicator from video 1)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 2. Long-Term Moving Averages (T3 simulation from video 3)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['trend_alignment'] = (df['sma_50'] > df['sma_200']).astype(int) # 1 if Bullish, 0 if Bearish
    
    # 3. Momentum & Volume (Trend strength confirmation)
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_20'] * 1.5).astype(int)
    
    # Target: Predict mid-term trend (will price be higher than current in 5 candles)
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    df = df.dropna()
    return df

def train_trend_model():
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
            df = prepare_trend_data(full_data_path)
            all_data.append(df)
            
    if not all_data:
        print("Error: No data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'macd', 'macd_signal', 'macd_hist', 'trend_alignment', 'volume_surge']
    X = combined_df[feature_columns]
    y = combined_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training Trend-Following AI Model...")
    # Use more conservative settings for macro-trends
    model = HistGradientBoostingClassifier(random_state=42, max_iter=200, max_depth=4, learning_rate=0.01)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTrend Model Accuracy: {accuracy * 100:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'trend_model.joblib')
    joblib.dump(model, model_path)
    print(f"Trend Model saved to {model_path}")

if __name__ == "__main__":
    train_trend_model()