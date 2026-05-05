"""
Hardware configuration — call configure() once at the top of every training/bot entry point.

Sets:
  - torch CPU thread count = all logical cores
  - torch TF32 + cuDNN benchmark for RTX 3080 Ti (Ampere, full throughput)
  - sklearn/joblib global n_jobs = -1 (all cores)
  - Darts checkpoint dir on D drive
"""
from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)


def configure(verbose: bool = True) -> dict:
    """Apply CPU + GPU optimisations. Returns a dict of what was set."""
    result: dict = {}

    # ── CPU threads ───────────────────────────────────────────────────────────
    n_cpu = os.cpu_count() or 1

    try:
        import torch
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))
        result['torch_threads'] = n_cpu
    except Exception as e:
        logger.debug("torch thread config skipped: %s", e)

    # ── GPU / CUDA optimisations ──────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            # TF32: ~3× faster matmul on Ampere (RTX 30xx) with <0.1% accuracy difference
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            # cuDNN benchmark: auto-selects fastest conv algorithm for current input shape
            torch.backends.cudnn.benchmark        = True
            torch.backends.cudnn.deterministic    = False  # must be False for benchmark
            gpu_name = torch.cuda.get_device_name(0)
            result['gpu'] = gpu_name
            result['tf32'] = True
            result['cudnn_benchmark'] = True
            if verbose:
                logger.info("[HW] GPU: %s  TF32=ON  cuDNN benchmark=ON", gpu_name)
        else:
            result['gpu'] = 'CPU only'
            if verbose:
                logger.info("[HW] No CUDA GPU — running on CPU (%d threads)", n_cpu)
    except Exception as e:
        logger.debug("GPU config skipped: %s", e)

    # ── sklearn global parallelism ────────────────────────────────────────────
    try:
        import sklearn
        sklearn.set_config(assume_finite=True)   # skip NaN checks (speed-up)
    except Exception:
        pass

    # Set joblib default backend so CalibratedClassifierCV cv-folds run in parallel
    try:
        import joblib
        # Patch the default n_jobs used by CalibratedClassifierCV when n_jobs not set
        os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(n_cpu))
    except Exception:
        pass

    if verbose:
        logger.info("[HW] CPU: %d logical cores  OMP=%s  MKL=%s",
                    n_cpu,
                    os.environ.get('OMP_NUM_THREADS', 'unset'),
                    os.environ.get('MKL_NUM_THREADS', 'unset'))

    result['cpu_cores'] = n_cpu
    return result


# ── VRAM brackets for TFT model sizing ───────────────────────────────────────
# Razer Blade budget: 8 GB VRAM, 4 GB reserved for QuestDB Docker → ≤7 GB usable.
# Offload heavier fold counts to PC2/PC3 workers (no Docker overhead there).
_VRAM_BRACKETS = [
    (24, {"batch_size": 128, "hidden_size": 128, "lstm_layers": 2}),
    (16, {"batch_size":  64, "hidden_size":  64, "lstm_layers": 2}),
    ( 8, {"batch_size":  32, "hidden_size":  64, "lstm_layers": 2}),  # Razer Blade
    ( 4, {"batch_size":  16, "hidden_size":  32, "lstm_layers": 1}),
    ( 0, {"batch_size":  16, "hidden_size":  32, "lstm_layers": 1}),  # CPU fallback
]

# Minimum free RAM (GB) to keep available for QuestDB Docker on the primary node.
QUESTDB_DOCKER_RESERVE_GB = 4.0


def get_tft_config(verbose: bool = False) -> dict:
    """Return safe TFT hyperparameters for the available VRAM.

    On the Razer Blade (8 GB VRAM, 4 GB Docker reservation) this returns
    batch_size=32 / hidden_size=64 / lstm_layers=2 — enough capacity without
    OOMing or starving QuestDB.  Workers with ≥16 GB VRAM receive larger configs
    so the orchestrator can offload heavy fold training to PC2/PC3.
    """
    vram_gb = 0.0
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            device = "gpu"
    except Exception:
        pass

    cfg: dict = {"batch_size": 16, "hidden_size": 32, "lstm_layers": 1}
    for threshold, params in _VRAM_BRACKETS:
        if vram_gb >= threshold:
            cfg = dict(params)
            break

    cfg["device"] = device
    cfg["vram_gb"] = round(vram_gb, 1)

    if verbose:
        logger.info(
            "[HW] TFT config: batch=%d hidden=%d lstm=%d  device=%s  VRAM=%.1f GB",
            cfg["batch_size"], cfg["hidden_size"], cfg["lstm_layers"], device, vram_gb,
        )
    return cfg
