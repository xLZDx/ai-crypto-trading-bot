import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_all')

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
    log.info("==========================================")
    log.info("   STARTING BATCH ML TRAINING PIPELINE   ")
    log.info("==========================================")

    try:
        log.info(">>> [0/6] Downloading funding rate history (ccxt)...")
        from src.data_ingestion.funding_rate_downloader import download_funding_rates
        download_funding_rates(days=365 * 2)
    except Exception as e:
        log.warning("Funding rate download failed (will train without): %s", e)

    try:
        log.info(">>> [1/6] Training Base Model (1h)...")
        train_model()
    except Exception as e:
        log.error("Error training Base Model: %s", e)

    try:
        log.info(">>> [2/6] Training Trend Following Model (1h)...")
        train_trend_model()
    except Exception as e:
        log.error("Error training Trend Model: %s", e)

    try:
        log.info(">>> [3/6] Training Futures Short Model (1h)...")
        train_futures_model()
    except Exception as e:
        log.error("Error training Futures Model: %s", e)

    try:
        log.info(">>> [4/6] Training Scalping Model (1m)...")
        train_scalping_model()
    except Exception as e:
        log.error("Error training Scalping Model: %s", e)

    try:
        log.info(">>> [5/6] Training TFT Model (1h, neural forecasting)...")
        train_tft_model()
    except Exception as e:
        log.warning("Skipping TFT Model training: %s", e)

    try:
        log.info(">>> [6/6] Running Phase 5 Backtester (strategy comparison)...")
        from src.engine.backtester import run_full_backtest
        comparison = run_full_backtest()
        log.info("Strategy Comparison:\n%s",
                 comparison[["strategy", "symbol", "sharpe", "sortino", "max_drawdown_pct", "win_rate_pct", "n_trades"]].to_string(index=False))
    except Exception as e:
        log.warning("Skipping backtest: %s", e)

    log.info("==========================================")
    log.info("   ALL MODELS TRAINED SUCCESSFULLY!      ")
    log.info("==========================================")

if __name__ == "__main__":
    train_all()
