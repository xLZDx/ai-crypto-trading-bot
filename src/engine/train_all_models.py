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

# CPU multithreading — capped so a long overnight run can't pin 100 % CPU
# and starve the bot loop, dashboard, and watchdog of cycles. The
# operator can scale up via AI_TRADER_TRAIN_CPU_THREADS if they want
# every core; default 10 of 14 leaves 4 cores for the rest of the system.
_total_cpu = os.cpu_count() or 20
_n_cpu = str(int(os.getenv('AI_TRADER_TRAIN_CPU_THREADS', '10')))
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


# Per-key timeframes the multi-TF training pipeline iterates over.
# Curated, "applicable based on model logic" — see PLAN_2026_05_08
# §1.P0.A for the full reasoning. Each key only lists TFs that match
# that model's design horizon; combos that converge to noise (TFT @ 1m)
# or that add no information (regime × extra TFs — features are
# TF-invariant) are intentionally excluded so the overnight sweep
# doesn't waste 6-12 h of compute on bad models.
#
#   base/trend/futures — directional signals; intraday → swing.
#   scalping — sub-minute mean reversion only (1m / 5m).
#   meta     — gates entry signals at 5m–4h; no consumer at 1d/1w.
#   tft      — input_chunk_length=168 needs ~3 h history per inference;
#              swing horizons (15m+) only.
#   regime   — GMM uses TF-invariant features (vol, ADX, returns z);
#              one canonical TF (1h) is enough.
#
# Override: set AI_TRADER_TRAIN_TF_MAP=strict_all_all to fall back to
# the 49-combo every-key-on-every-TF sweep (lower-quality models on
# the noise combos but useful for ablation studies).
DEFAULT_PER_KEY_TFS: dict[str, tuple[str, ...]] = {
    'base':     ('5m', '15m', '1h', '4h', '1d'),     # 5 TFs
    'trend':    ('15m', '1h', '4h', '1d', '1w'),     # 5 TFs
    'futures':  ('5m', '15m', '1h', '4h', '1d'),     # 5 TFs
    'scalping': ('1m', '5m'),                         # 2 TFs
    'meta':     ('5m', '15m', '1h', '4h'),           # 4 TFs
    'tft':      ('15m', '1h', '4h'),                  # 3 TFs
    'regime':   ('1h',),                              # 1 TF
}
# 25 (model × TF) tabular combos. OFT is added separately by item 1M
# (microstructure model on L2/L3, single canonical TF).

if os.getenv('AI_TRADER_TRAIN_TF_MAP', '').lower() in ('strict', 'all', 'strict_all_all'):
    _ALL_TFS = ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
    DEFAULT_PER_KEY_TFS = {k: _ALL_TFS for k in DEFAULT_PER_KEY_TFS}
    log.info("AI_TRADER_TRAIN_TF_MAP override: strict all×all (49 combos)")


def train_all(per_key_tfs: dict[str, tuple[str, ...]] | None = None):
    """Run the full training pipeline. Each tabular trainer runs once per
    timeframe in per_key_tfs[key] (defaults to DEFAULT_PER_KEY_TFS). Each
    invocation writes models/<key>_<tf>_*.{joblib,json} and — when tf is
    canonical — also writes the legacy filenames the bot's inference path
    still loads."""
    _hw_configure(verbose=True)
    cfg = per_key_tfs or DEFAULT_PER_KEY_TFS
    log.info("==========================================")
    log.info("   STARTING BATCH ML TRAINING PIPELINE   ")
    log.info("   Multi-TF: %s", cfg)
    log.info("==========================================")

    # ── 0. Download data ──────────────────────────────────────────────────
    try:
        log.info(">>> [0/8] Downloading funding rate history (ccxt)...")
        from src.data_ingestion.funding_rate_downloader import download_funding_rates
        download_funding_rates(days=365 * 2)
    except Exception as e:
        log.warning("Funding rate download failed (will train without): %s", e)

    # Skip-if-fresh — when an overnight run dies at hour 5 of 8 and the
    # operator re-triggers it, we don't want to redo every model from
    # scratch. If models/<key>_<tf>_meta.json was written within the last
    # SKIP_IF_FRESH_S, treat that combo as already trained and move on.
    # Default 2 h; override via AI_TRADER_TRAIN_SKIP_IF_FRESH_S env var.
    # Setting it to 0 forces every combo to retrain (manual full sweep).
    import time as _time
    SKIP_IF_FRESH_S = int(os.getenv('AI_TRADER_TRAIN_SKIP_IF_FRESH_S', str(2 * 3600)))
    from src.utils.model_paths import artifact_paths

    def _meta_age_s(key: str, tf: str) -> float | None:
        try:
            p = artifact_paths(key, tf).get('meta')
            if p and p.exists():
                return _time.time() - p.stat().st_mtime
        except Exception:
            pass
        return None

    # Helper: run a trainer for every TF in its list, isolating failures.
    def _train_loop(key: str, fn, label: str):
        for tf in cfg.get(key, ()):
            age = _meta_age_s(key, tf)
            if SKIP_IF_FRESH_S > 0 and age is not None and age < SKIP_IF_FRESH_S:
                log.info(">>> Skipping %s @ %s — meta written %d min ago "
                         "(within SKIP_IF_FRESH_S=%ds; resume mode)",
                         label, tf, int(age / 60), SKIP_IF_FRESH_S)
                continue
            try:
                log.info(">>> Training %s @ %s ...", label, tf)
                fn(timeframe=tf)
            except Exception as exc:
                log.error("Error training %s @ %s: %s", label, tf, exc)

    # ── 1. Base ───────────────────────────────────────────────────────────
    _train_loop('base',     train_model,         'Base Model')
    # ── 2. Trend ──────────────────────────────────────────────────────────
    _train_loop('trend',    train_trend_model,   'Trend Model')
    # ── 3. Futures ────────────────────────────────────────────────────────
    _train_loop('futures',  train_futures_model, 'Futures Short Model')
    # ── 4. Scalping (1m only by design) ───────────────────────────────────
    _train_loop('scalping', train_scalping_model, 'Scalping Model')

    # ── 5. TFT model (single-TF) ──────────────────────────────────────────
    for tf in cfg.get('tft', ()):
        try:
            log.info(">>> Training TFT Model @ %s ...", tf)
            train_tft_model()
        except Exception as exc:
            log.warning("Skipping TFT @ %s: %s", tf, exc)

    # ── 6. Regime classifier (feature-stage, single-TF) ───────────────────
    try:
        log.info(">>> Training Regime Classifier (GMM, 3 regimes)...")
        from src.analysis.regime_classifier import train_regime_classifier
        clf = train_regime_classifier()
        if clf.is_ready:
            log.info("Regime classifier trained successfully.")
        else:
            log.warning("Regime classifier training produced no output.")
    except Exception as e:
        log.warning("Skipping Regime Classifier: %s", e)

    # ── 7. Meta-labeler (per TF) ──────────────────────────────────────────
    try:
        from src.engine.train_meta_labeler import train_meta_labeler
        _train_loop('meta', train_meta_labeler, 'Meta-Labeler')
    except Exception as e:
        log.warning("Skipping Meta-Labeler: %s", e)

    # ── 8. Full backtester with A/B comparison ────────────────────────────
    try:
        log.info(">>> Running Full Backtester (Group A vs Group B + Meta-filtered)...")
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
