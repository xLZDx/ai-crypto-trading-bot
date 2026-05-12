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


# 2026-05-10 fix — operator on the same machine had 6 zombie
# train_all_models.py processes competing for CPU. Each launch (CLI or
# dashboard "Retrain ALL") spawned without checking whether another was
# already running. Concurrency lock at process level — refuses second
# launch unless --force is passed.
#
# Lock file: data/train_all_models.lock — JSON {pid, started_iso, host}.
# Stale-lock detection: if pid not alive, lock is reclaimed automatically.
_LOCK_PATH = os.path.join(project_root, 'data', 'train_all_models.lock')
_CURRENT_PATH = os.path.join(project_root, 'data', 'training_current.json')


def _acquire_run_lock(force: bool = False) -> bool:
    """Acquire the single-instance lock. Returns True on success.
    Returns False (and logs reason) if another live instance holds it.
    Auto-reclaims stale locks (pid not alive)."""
    import json as _json, datetime as _dt, socket as _sock
    if os.path.exists(_LOCK_PATH):
        try:
            with open(_LOCK_PATH, 'r', encoding='utf-8') as f:
                prev = _json.load(f)
        except Exception:
            prev = {}
        prev_pid = int(prev.get('pid', 0))
        if prev_pid:
            alive = False
            try:
                import psutil
                alive = psutil.pid_exists(prev_pid)
                if alive:
                    p = psutil.Process(prev_pid)
                    alive = ('python' in (p.name() or '').lower()
                             and 'train_all_models' in ' '.join(p.cmdline() or []))
            except Exception:
                pass
            if alive and not force:
                log.error("Another train_all_models.py is already running "
                          "(pid=%d, started=%s). Pass --force to override "
                          "or kill that process first.",
                          prev_pid, prev.get('started_iso', '?'))
                return False
            if alive and force:
                log.warning("Another train_all_models.py is running (pid=%d) "
                            "but --force given; proceeding in parallel "
                            "(both will compete for CPU/GPU).",
                            prev_pid)
            else:
                log.info("Reclaiming stale lock from dead pid=%d", prev_pid)
    payload = {
        'pid':         os.getpid(),
        'started_iso': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'host':        _sock.gethostname(),
    }
    try:
        os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
        # Phase A12 (2026-05-12): atomic write via safe_json. The
        # previous raw open(,'w') had a race where a concurrent
        # _acquire_run_lock / _release_run_lock could see a half-
        # written file and produce a corrupted lock state. safe_json
        # uses filelock + write-temp-then-rename to make the write
        # atomic from any reader's point of view.
        from src.utils.safe_json import write_json as _safe_write
        _safe_write(_LOCK_PATH, payload)
    except Exception as exc:
        log.warning("Could not write lock file %s: %s — proceeding anyway",
                    _LOCK_PATH, exc)
    return True


def _release_run_lock() -> None:
    """Best-effort lock release on graceful exit. Stale locks from
    crashes are auto-reclaimed by _acquire_run_lock on next launch."""
    try:
        if os.path.exists(_LOCK_PATH):
            # Phase A12 (2026-05-12): read via safe_json so a concurrent
            # write in progress doesn't yield a torn/incomplete dict.
            from src.utils.safe_json import read_json as _safe_read
            prev = _safe_read(_LOCK_PATH, default={}) or {}
            if int(prev.get('pid', 0)) == os.getpid():
                os.remove(_LOCK_PATH)
    except Exception:
        pass


def _set_current(model_key: str | None, tf: str | None,
                 label: str | None, *, status: str = 'running') -> None:
    """Write the actively-training model to data/training_current.json.
    Dashboard's orphan detector reads this file and overrides the
    'orphan = model=all' fan-out with the actual current model.

    Pass model_key=None at end-of-run to clear the file."""
    import json as _json, datetime as _dt
    try:
        if model_key is None:
            if os.path.exists(_CURRENT_PATH):
                os.remove(_CURRENT_PATH)
            return
        payload = {
            'model_key':   model_key,
            'current_tf':  tf,
            'label':       label or model_key,
            'status':      status,
            'parent_pid':  os.getpid(),
            'updated_iso': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(_CURRENT_PATH), exist_ok=True)
        # Phase A12 (2026-05-12): atomic write via safe_json — same
        # race-condition fix as _acquire_run_lock above. The dashboard
        # orphan detector reads this file frequently; a half-written
        # state would surface as a transient bogus orphan banner.
        from src.utils.safe_json import write_json as _safe_write
        _safe_write(_CURRENT_PATH, payload)
    except Exception:
        # Best-effort — don't let a status-file failure crash training.
        pass


def train_all(per_key_tfs: dict[str, tuple[str, ...]] | None = None,
              *, force: bool = False):
    """Run the full training pipeline. Each tabular trainer runs once per
    timeframe in per_key_tfs[key] (defaults to DEFAULT_PER_KEY_TFS). Each
    invocation writes models/<key>_<tf>_*.{joblib,json} and — when tf is
    canonical — also writes the legacy filenames the bot's inference path
    still loads.

    force=True bypasses the single-instance lock — only use it when you
    deliberately want two parallel runs (rare; usually a mistake)."""
    if not _acquire_run_lock(force=force):
        return  # already running, bail out cleanly
    try:
        _train_all_inner(per_key_tfs)
    finally:
        # Always clear the current-model state and release the lock,
        # even if the pipeline raised. Stale lock from a crash IS
        # auto-reclaimed on next launch via _acquire_run_lock, but
        # cleaner to release on graceful exit.
        _set_current(None, None, None)
        _release_run_lock()


def _train_all_inner(per_key_tfs: dict[str, tuple[str, ...]] | None = None):
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
    #
    # Resolution order (v3.1 fix 2026-05-09 — operator wanted today's work
    # preserved across crashes / watchdog respawns):
    #   1. data/training_config.json `skip_if_fresh_s` field — survives
    #      restarts, can be edited without re-spawning anything.
    #   2. AI_TRADER_TRAIN_SKIP_IF_FRESH_S env var.
    #   3. Default 48 h (was 2 h pre-fix; bumped because watchdog-triggered
    #      respawns from the dashboard don't inherit env vars set when
    #      I manually launched the orchestrator with a custom value, so
    #      a crash 13+ h into an overnight sweep would silently re-train
    #      everything completed earlier in the day with the old default).
    # Setting it to 0 forces every combo to retrain (manual full sweep).
    import time as _time
    import json as _json_skip
    _cfg_skip_s = None
    try:
        _cfg_path = os.path.join(project_root, 'data', 'training_config.json')
        if os.path.exists(_cfg_path):
            with open(_cfg_path, 'r', encoding='utf-8') as _f:
                _cfg = _json_skip.load(_f) or {}
            v = _cfg.get('skip_if_fresh_s')
            if isinstance(v, (int, float)) and v >= 0:
                _cfg_skip_s = int(v)
    except Exception:
        _cfg_skip_s = None
    SKIP_IF_FRESH_S = (_cfg_skip_s if _cfg_skip_s is not None
                       else int(os.getenv('AI_TRADER_TRAIN_SKIP_IF_FRESH_S',
                                          str(48 * 3600))))
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
                _set_current(key, tf, label)
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

    # ── 5. TFT model (per-TF; v3.1 step 4 fix passes the TF through) ──────
    for tf in cfg.get('tft', ()):
        age = _meta_age_s('tft', tf)
        if SKIP_IF_FRESH_S > 0 and age is not None and age < SKIP_IF_FRESH_S:
            log.info(">>> Skipping TFT @ %s — meta written %d min ago", tf, int(age / 60))
            continue
        try:
            log.info(">>> Training TFT Model @ %s ...", tf)
            _set_current('tft', tf, 'TFT Model')
            train_tft_model(timeframe=tf)
        except Exception as exc:
            log.warning("Skipping TFT @ %s: %s", tf, exc)

    # ── 5½. OFT (Microstructure) — v3.1 step 8 / 1M ───────────────────────
    # OFT trains on L2/L3 order-book microstructure events; one
    # canonical TF (1m) per symbol since higher TFs lose the
    # microstructure detail. Wrapped in try/except per the rest of
    # the sweep — a failure here can't crash later trainers. The
    # canonical symbol set keeps the sweep bounded; operator can
    # extend with AI_TRADER_OFT_SYMBOLS env var.
    try:
        from src.training.joint_oft_rl import train_oft
        oft_symbols = os.getenv('AI_TRADER_OFT_SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT').split(',')
        oft_symbols = [s.strip() for s in oft_symbols if s.strip()]
        oft_tf = '1m'
        for sym in oft_symbols:
            try:
                log.info(">>> Training OFT (Microstructure) @ %s/%s ...", sym, oft_tf)
                _set_current('oft', oft_tf, f'OFT ({sym})')
                train_oft(sym, oft_tf)
            except Exception as exc:
                log.warning("Skipping OFT %s/%s: %s", sym, oft_tf, exc)
    except ImportError as exc:
        log.warning("Skipping OFT entirely (joint_oft_rl unavailable): %s", exc)

    # ── 6. Regime classifier (feature-stage, single-TF) ───────────────────
    try:
        log.info(">>> Training Regime Classifier (GMM, 3 regimes)...")
        _set_current('regime', '1h', 'Regime Classifier')
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
    import argparse
    _parser = argparse.ArgumentParser(description='Run the full training pipeline')
    _parser.add_argument('--force', action='store_true',
                         help='Bypass single-instance lock (allow parallel runs).')
    _args, _unknown = _parser.parse_known_args()
    train_all(force=_args.force)
