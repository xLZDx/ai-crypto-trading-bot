import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_data(filepath):
    print(f"Loading data from {filepath}...")
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime and sort chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print("Engineering features (indicators)...")
    # 1. Price return (percentage change)
    df['return'] = df['close'].pct_change()
    # 2. Simple Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    # 3. Volatility (rolling standard deviation)
    df['volatility'] = df['return'].rolling(window=7).std()
    
    # 4. Distance from price to moving averages (percentage)
    df['dist_sma_7'] = df['close'] / df['sma_7'] - 1
    df['dist_sma_30'] = df['close'] / df['sma_30'] - 1
    
    # 5. Classic RSI (14)
    delta = df['close'].diff()
    df['rsi_14'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(com=13, adjust=False).mean() / (-1 * delta.clip(upper=0)).ewm(com=13, adjust=False).mean())))
    
    # 6. MACD (Trend Momentum)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 7. Volume Momentum (Volume strength relative to 14-day average)
    df['vol_sma_14'] = df['volume'].rolling(window=14).mean()
    df['volume_momentum'] = df['volume'] / df['vol_sma_14']
    
    # 8. Stochastic Oscillator (Where we are in the 14-day range)
    df['high_14'] = df['high'].rolling(window=14).max()
    df['low_14'] = df['low'].rolling(window=14).min()
    high_low_diff = (df['high_14'] - df['low_14']).replace(0, 0.0001) # Protection against division by zero
    df['stoch_k'] = (df['close'] - df['low_14']) / high_low_diff * 100
    
    # 9. Inertia (Return for yesterday and the day before)
    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    
    # 10. Additional inertial and volatility features
    df['return_lag3'] = df['return'].shift(3)
    df['return_lag5'] = df['return'].shift(5)
    df['atr_pct'] = (df['high'] - df['low']) / df['close']
    
    if 'taker_buy_base' in df.columns:
        df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, 0.0001)
    else:
        df['taker_buy_ratio'] = 0.5
        
    if 'trades_count' in df.columns:
        df['avg_trade_size'] = df['volume'] / df['trades_count'].replace(0, 1)
    else:
        df['avg_trade_size'] = 0.0

    # Target Variable: 1 if the NEXT candle's close is higher than CURRENT close, else 0
    # We shift by -1 to peek at the next row's closing price
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop rows with NaN values created by rolling windows and shifting
    df = df.dropna()
    
    return df

def train_model():
    # Construct absolute path to the data downloaded by historical_backfill.py
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    wl_path = os.path.join(base_dir, 'data', 'watchlist.json')
    if os.path.exists(wl_path):
        import json
        with open(wl_path, 'r') as f:
            symbols = [s.replace('/', '_') for s in json.load(f)]
    else:
        symbols = ['BTC_USDT', 'SOL_USDT', 'ADA_USDT']
        
    timeframe = '1h' # USING 1h, AS THE BOT TRADES ON HOURLY CANDLES!
    
    all_data = []
    for sym in symbols:
        full_data_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_{timeframe}.csv.gz')
        if not os.path.exists(full_data_path):
            print(f"Warning: Data for {sym} not found at {full_data_path}. Auto-downloading...")
            import sys
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from src.data_ingestion.historical_backfill import backfill_history
            backfill_history(symbol=sym.replace('_', '/'), timeframe=timeframe, days=6*365)
            
        if os.path.exists(full_data_path):
            print(f"Processing {sym}...")
            df = prepare_data(full_data_path)
            all_data.append(df)
            
    if not all_data:
        print("Error: No data found to train the model even after attempted download.")
        return
        
    # Combine data from all coins into one huge dataset
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Define the features (X) the model will use to make predictions
    feature_columns = ['return', 'volatility', 'dist_sma_7', 'dist_sma_30', 'rsi_14', 'macd', 'macd_hist', 'volume_momentum', 'stoch_k', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5', 'atr_pct', 'taker_buy_ratio', 'avg_trade_size']
    X = combined_df[feature_columns]
    
    # Define the target (y) the model is trying to predict
    y = combined_df['target']
    
    print("Splitting data into training and testing sets...")
    # IMPORTANT: shuffle=False prevents "look-ahead bias" in time-series data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training HistGradientBoosting Classifier...")
    # Gradient boosting works much better with complex financial data than random forest
    model = HistGradientBoostingClassifier(random_state=42, max_iter=300, max_depth=5, learning_rate=0.05, l2_regularization=0.1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Save the trained model to disk
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'btc_rf_model.joblib')
    
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to {model_path}")
    
    # Save accuracy metadata for the dashboard
    meta_path = os.path.join(models_dir, 'model_meta.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump({"accuracy": accuracy * 100}, f)

if __name__ == "__main__":
    train_model()