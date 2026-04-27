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

from src.analysis.feature_engineering import add_rsi, add_roc, add_time_features


def prepare_futures_data(filepath):
    print(f"Loading data for Futures (Shorting) Pipeline from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    df['return'] = df['close'].pct_change()
    df = add_rsi(df, 14)
    df = add_roc(df, [5])
    df = add_time_features(df)

    df['low_30'] = df['low'].rolling(30).min()
    df['dist_to_support'] = (df['close'] - df['low_30']) / df['close']

    df['vol_sma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_drop'] = (df['volume'] < df['vol_sma_7'] * 0.7).astype(int)

    # Target: price drops >0.5% within 3 candles
    df['target_short'] = (df['close'].shift(-3) < df['close'] * 0.995).astype(int)
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
        archive_path = os.path.join(base_dir, 'data', 'raw', f'{sym}_spot_1h.csv.gz')
        if not os.path.exists(full_data_path) and os.path.exists(archive_path):
            full_data_path = archive_path
            print(f"  Using archive data for {sym}: {archive_path}")
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
    
    feature_columns = ['return', 'rsi_14', 'dist_to_support', 'volume_drop', 'hour', 'roc_5']
    X = combined_df[feature_columns]
    y = combined_df['target_short']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    print("Training Futures Shorting AI Model...")
    # Train model to find rare but strong dumps
    model = HistGradientBoostingClassifier(random_state=42, max_iter=400, max_depth=6, learning_rate=0.03, l2_regularization=0.2, early_stopping=True, class_weight='balanced')
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    # Target 1 means DROP (Short), Target 0 means UP/HOLD (Long)
    short_acc = report.get('1', {}).get('precision', 0.0) * 100
    long_acc = report.get('0', {}).get('precision', 0.0) * 100

    print(f"\nFutures Short Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Long (UP) Precision: {long_acc:.2f}% | Short (DOWN) Precision: {short_acc:.2f}%")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'futures_short_model.joblib')
    joblib.dump(model, model_path)
    print(f"Futures Model saved to {model_path}")
    
    meta_path = os.path.join(models_dir, 'futures_short_model_meta.json')
    import json
    from datetime import datetime, timezone
    with open(meta_path, 'w') as f:
        json.dump({"accuracy": accuracy * 100, "long_accuracy": long_acc, "short_accuracy": short_acc,
                   "last_trained": datetime.now(timezone.utc).isoformat()}, f)

if __name__ == "__main__":
    train_futures_model()