import os
import sys
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.analysis.feature_engineering import (
    add_taker_and_trade_features, add_rsi, add_macd,
    add_bollinger_bands, add_roc, add_time_features
)


def prepare_scalping_data(filepath):
    print(f"Loading data for Scalping Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    df['return'] = df['close'].pct_change()
    df = add_rsi(df, period=7, col_name='rsi_7')
    df = add_macd(df, fast=5, slow=13, signal=3, prefix='')
    df.rename(columns={'macd': 'macd_fast', 'macd_signal': 'macd_fast_signal', 'macd_hist': 'macd_fast_hist'}, errors='ignore', inplace=True)
    df = add_bollinger_bands(df, window=10)
    df.rename(columns={'bb_pb': 'bb_pb', 'bb_upper': 'bb_upper', 'bb_lower': 'bb_lower'}, inplace=True)
    df = add_roc(df, [3, 5, 10])
    df = add_time_features(df)
    df = add_taker_and_trade_features(df)

    df['vol_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_surge'] = (df['volume'] > df['vol_sma_5'] * 2.0).astype(int)

    df['low_15'] = df['low'].rolling(15).min()
    df['dist_to_micro_supp'] = (df['close'] - df['low_15']) / df['close']

    df['target_scalp'] = (df['close'].shift(-5) > df['close']).astype(int)
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
            # INCREASED MEMORY LIMIT: Take the last 500,000 candles (~1 year)
            # Utilizes the bulk data archive without causing out-of-memory crashes
            df = df.tail(500000)
            all_data.append(df)
            
    if not all_data:
        print("Error: No 1m data found even after attempted download.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    feature_columns = ['return', 'rsi_7', 'macd_fast', 'volume_surge', 'dist_to_micro_supp', 'taker_buy_ratio', 'avg_trade_size', 'hour', 'roc_3', 'roc_5', 'roc_10', 'bb_pb']
    X = combined_df[feature_columns]
    y = combined_df['target_scalp']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    print("Training Scalping AI Model...")
    model = HistGradientBoostingClassifier(random_state=42, max_iter=400, max_depth=5, learning_rate=0.05, early_stopping=True, class_weight='balanced')
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    long_acc = report.get('1', {}).get('precision', 0.0) * 100
    short_acc = report.get('0', {}).get('precision', 0.0) * 100

    print(f"\nScalping Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Long (UP) Precision: {long_acc:.2f}% | Short (DOWN) Precision: {short_acc:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'scalping_model.joblib')
    joblib.dump(model, model_path)
    print(f"Scalping Model saved to {model_path}")
    
    meta_path = os.path.join(models_dir, 'scalping_model_meta.json')
    import json
    with open(meta_path, 'w') as f:
        json.dump({"accuracy": accuracy * 100, "long_accuracy": long_acc, "short_accuracy": short_acc}, f)

if __name__ == "__main__":
    train_scalping_model()