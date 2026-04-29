import sys
import os
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Redirect all caches to D: drive before any imports that write to C:
_cache = os.path.join(project_root, 'data', 'cache')
for _sub in ('temp', 'torch', 'huggingface', 'cuda', 'numba', 'pip'):
    os.makedirs(os.path.join(_cache, _sub), exist_ok=True)
os.environ.setdefault('TMP',                  os.path.join(_cache, 'temp'))
os.environ.setdefault('TEMP',                 os.path.join(_cache, 'temp'))
os.environ.setdefault('HF_HOME',              os.path.join(_cache, 'huggingface'))
os.environ.setdefault('TORCH_HOME',           os.path.join(_cache, 'torch'))
os.environ.setdefault('JOBLIB_TEMP_FOLDER',   os.path.join(_cache, 'temp'))
os.environ.setdefault('CUDA_CACHE_PATH',      os.path.join(_cache, 'cuda'))
os.environ.setdefault('NUMBA_CACHE_DIR',      os.path.join(_cache, 'numba'))
os.environ.setdefault('MPLCONFIGDIR',         os.path.join(_cache, 'matplotlib'))

# CPU multithreading — use every logical core
_n_cpu = str(os.cpu_count() or 20)
os.environ.setdefault('OMP_NUM_THREADS',      _n_cpu)
os.environ.setdefault('MKL_NUM_THREADS',      _n_cpu)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _n_cpu)
os.environ.setdefault('NUMEXPR_NUM_THREADS',  _n_cpu)
os.environ.setdefault('LOKY_MAX_CPU_COUNT',   _n_cpu)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('train_all')

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.train_model import train_model
from src.engine.train_trend_model import train_trend_model
from src.engine.train_futures_model import train_futures_model
from src.engine.train_scalping_model import train_scalping_model
from src.engine.train_tft_model import train_tft_model
from src.utils.hw_config import configure as _hw_configure


def train_all():
    _hw_configure(verbose=True)
    log.info("==========================================")
    log.info("   STARTING BATCH ML TRAINING PIPELINE   ")
    log.info("==========================================")

    # ── 0. Download data ──────────────────────────────────────────────────
    try:
        log.info(">>> [0/8] Downloading funding rate history (ccxt)...")
        from src.data_ingestion.funding_rate_downloader import download_funding_rates
        download_funding_rates(days=365 * 2)
    except Exception as e:
        log.warning("Funding rate download failed (will train without): %s", e)

    # ── 1. Base model ─────────────────────────────────────────────────────
    try:
        log.info(">>> [1/8] Training Base Model (1h, Triple Barrier, Walk-Forward, Calibrated)...")
        train_model()
    except Exception as e:
        log.error("Error training Base Model: %s", e)

    # ── 2. Trend model ────────────────────────────────────────────────────
    try:
        log.info(">>> [2/8] Training Trend Model (1h, Triple Barrier, Donchian+Keltner)...")
        train_trend_model()
    except Exception as e:
        log.error("Error training Trend Model: %s", e)

    # ── 3. Futures model ──────────────────────────────────────────────────
    try:
        log.info(">>> [3/8] Training Futures Short Model (1h, Triple Barrier, Funding Z-score)...")
        train_futures_model()
    except Exception as e:
        log.error("Error training Futures Model: %s", e)

    # ── 4. Scalping model ─────────────────────────────────────────────────
    try:
        log.info(">>> [4/8] Training Scalping Model (1m, Triple Barrier 5-bar, OFI+VWAP)...")
        train_scalping_model()
    except Exception as e:
        log.error("Error training Scalping Model: %s", e)

    # ── 5. TFT model ──────────────────────────────────────────────────────
    try:
        log.info(">>> [5/8] Training TFT Model (1h, neural probabilistic forecasting)...")
        train_tft_model()
    except Exception as e:
        log.warning("Skipping TFT Model training (optional): %s", e)

    # ── 6. Regime classifier ──────────────────────────────────────────────
    try:
        log.info(">>> [6/8] Training Regime Classifier (GMM, 3 regimes: ranging/trending/volatile)...")
        from src.analysis.regime_classifier import train_regime_classifier
        clf = train_regime_classifier()
        if clf.is_ready:
            log.info("Regime classifier trained successfully.")
        else:
            log.warning("Regime classifier training produced no output.")
    except Exception as e:
        log.warning("Skipping Regime Classifier: %s", e)

    # ── 7. Meta-labeler ───────────────────────────────────────────────────
    try:
        log.info(">>> [7/8] Training Meta-Labeler (second-pilot signal filter)...")
        from src.engine.train_meta_labeler import train_meta_labeler
        train_meta_labeler()
    except Exception as e:
        log.warning("Skipping Meta-Labeler training: %s", e)

    # ── 8. Full backtester with A/B comparison ────────────────────────────
    try:
        log.info(">>> [8/8] Running Full Backtester (Group A vs Group B + Meta-filtered)...")
        from src.engine.backtester import run_full_backtest
        comparison = run_full_backtest()
        if not comparison.empty:
            cols = ["strategy", "symbol", "sharpe", "sortino", "max_drawdown_pct",
                    "win_rate_pct", "n_trades"]
            available = [c for c in cols if c in comparison.columns]
            log.info("Strategy Comparison:\n%s", comparison[available].to_string(index=False))
        log.info("A/B comparison saved to data/backtest/ab_comparison.json")
    except Exception as e:
        log.warning("Skipping backtest: %s", e)

    log.info("==========================================")
    log.info("   ALL MODELS TRAINED SUCCESSFULLY!      ")
    log.info("==========================================")


if __name__ == "__main__":
    train_all()
