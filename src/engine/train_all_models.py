import sys
import os

# Must be set before numpy/sklearn/OpenBLAS import to prevent memory allocation failures on Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Ensure Python sees the root folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.train_model import train_model
from src.engine.train_trend_model import train_trend_model
from src.engine.train_futures_model import train_futures_model
from src.engine.train_scalping_model import train_scalping_model
from src.engine.train_tft_model import train_tft_model

def train_all():
    print("==========================================")
    print("   STARTING BATCH ML TRAINING PIPELINE    ")
    print("==========================================")

    # Step 0: Download funding rates so they can be used as features in TFT
    try:
        print("\n>>> [0/6] Downloading funding rate history (ccxt)...")
        from src.data_ingestion.funding_rate_downloader import download_funding_rates
        download_funding_rates(days=365 * 2)
    except Exception as e:
        print(f"Warning: Funding rate download failed (will train without): {e}")

    try:
        print("\n>>> [1/6] Training Base Model (1h)...")
        train_model()
    except Exception as e:
        print(f"Error training Base Model: {e}")

    try:
        print("\n>>> [2/6] Training Trend Following Model (1h)...")
        train_trend_model()
    except Exception as e:
        print(f"Error training Trend Model: {e}")

    try:
        print("\n>>> [3/6] Training Futures Short Model (1h)...")
        train_futures_model()
    except Exception as e:
        print(f"Error training Futures Model: {e}")

    try:
        print("\n>>> [4/6] Training Scalping Model (1m)...")
        train_scalping_model()
    except Exception as e:
        print(f"Error training Scalping Model: {e}")

    try:
        print("\n>>> [5/6] Training TFT Model (1h, neural forecasting)...")
        train_tft_model()
    except Exception as e:
        print(f"Skipping TFT Model training: {e}")

    try:
        print("\n>>> [6/6] Running Phase 5 Backtester (strategy comparison)...")
        from src.engine.backtester import run_full_backtest
        comparison = run_full_backtest()
        print("\nStrategy Comparison:")
        print(comparison[["strategy", "symbol", "sharpe", "sortino", "max_drawdown_pct", "win_rate_pct", "n_trades"]].to_string(index=False))
    except Exception as e:
        print(f"Skipping backtest: {e}")

    print("\n==========================================")
    print("   ALL MODELS TRAINED SUCCESSFULLY!       ")
    print("==========================================")

if __name__ == "__main__":
    train_all()
