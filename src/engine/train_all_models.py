import sys
import os

# Ensure Python sees the root folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.train_model import train_model
from src.engine.train_trend_model import train_trend_model
from src.engine.train_futures_model import train_futures_model
from src.engine.train_scalping_model import train_scalping_model

def train_all():
    print("==========================================")
    print("   STARTING BATCH ML TRAINING PIPELINE    ")
    print("==========================================")

    try:
        print("\n>>> [1/4] Training Base Model (1h)...")
        train_model()
    except Exception as e:
        print(f"Error training Base Model: {e}")

    try:
        print("\n>>> [2/4] Training Trend Following Model (1h)...")
        train_trend_model()
    except Exception as e:
        print(f"Error training Trend Model: {e}")

    try:
        print("\n>>> [3/4] Training Futures Short Model (1h)...")
        train_futures_model()
    except Exception as e:
        print(f"Error training Futures Model: {e}")

    try:
        print("\n>>> [4/4] Training Scalping Model (1m)...")
        train_scalping_model()
    except Exception as e:
        print(f"Error training Scalping Model: {e}")

    print("\n==========================================")
    print("   ALL MODELS TRAINED SUCCESSFULLY!       ")
    print("==========================================")

if __name__ == "__main__":
    train_all()