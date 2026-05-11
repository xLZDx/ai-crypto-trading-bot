"""
Training Worker Node — run this on any laptop in the local network.

What it does:
  1. Auto-detects hardware (GPU/CPU, VRAM, RAM)
  2. Registers with the master orchestrator
  3. Polls for tasks and executes model training jobs
  4. Writes trained model files and metrics back to the master's shared folder (UNC path)
     or uploads via REST if no shared drive is configured

Usage:
  python -m src.training.distributed.worker --master http://192.168.1.100:7700
  python -m src.training.distributed.worker --master http://192.168.1.100:7700 --name "RTX2800-PC"

Requirements (auto-installed on first run):
  pip install flask requests psutil torch --index-url https://download.pytorch.org/whl/cu118
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("worker")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WORKER_PORT = 7701          # HTTP port this worker listens on
POLL_SEC    = 10            # how often to poll master for tasks
HEARTBEAT_SEC = 15          # worker-registration heartbeat interval
# 2026-05-11 — task-level heartbeat. Posts status="heartbeat" to the
# orchestrator every TASK_HEARTBEAT_S so `last_update_at` stays fresh
# for the full duration of long-running trainers (Darts/Lightning's TFT
# reports progress via tqdm only, never calling back natively). The
# cluster watchdog's 5-min stale-window gate stops mis-firing on
# actively-progressing neural runs as a result.
TASK_HEARTBEAT_S = 60


# ─── Dependency bootstrap ────────────────────────────────────────────────────

def _ensure_deps() -> None:
    """Install required packages if missing."""
    needed = {
        "flask":   "flask",
        "requests": "requests",
        "psutil":  "psutil",
        "torch":   None,     # handled separately (CUDA version)
    }
    for pkg, install_name in needed.items():
        try:
            __import__(pkg)
        except ImportError:
            if install_name:
                logger.info("Installing %s…", install_name)
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name, "-q"])

    # Install PyTorch with CUDA if GPU present and torch not installed
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA not available with current torch — consider reinstalling with CUDA wheels")
    except ImportError:
        logger.info("Installing PyTorch (CUDA 11.8)…")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118", "-q"
        ])


# ─── Hardware detection ──────────────────────────────────────────────────────

def _detect_hardware() -> dict:
    import psutil
    info = {
        "hostname":       socket.gethostname(),
        "ip":             _local_ip(),
        "cpu_cores":      psutil.cpu_count(logical=False) or 1,
        "ram_gb":         round(psutil.virtual_memory().total / 1e9, 1),
        "gpu_name":       "CPU only",
        "gpu_vram_gb":    0.0,
        "cuda_available": False,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"]       = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"]    = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return info


def _local_ip() -> str:
    """Return 192.168.0.x LAN IP if available, otherwise any non-loopback IP."""
    try:
        import psutil
        for iface_addrs in psutil.net_if_addrs().values():
            for addr in iface_addrs:
                if addr.family == socket.AF_INET and addr.address.startswith("192.168.0."):
                    return addr.address
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ─── Phase 93 — live load sampling for /health + heartbeat ───────────────────
#
# Why this exists: master_agent at the time of writing only knew that a
# worker had registered — not whether its CPU/GPU were actually doing
# useful work. During the 2026-05-10 sweep we burned hours staring at
# nvidia-smi on the wrong PC because the dashboard had no per-worker
# load number. Putting cpu_percent/gpu_percent on every heartbeat closes
# that visibility gap without requiring a side-channel monitoring tool.

def _sample_live_load() -> dict:
    """Snapshot current CPU + GPU utilisation on this host.

    Returns:
      cpu_percent      — system-wide CPU % since the last sample
                          (psutil; 15-s window via the heartbeat cadence)
      gpu_percent      — GPU utilisation % from `nvidia-smi`, or None if
                          nvidia-smi missing/CPU-only host
      gpu_mem_used_mb  — VRAM in use (MiB), or None
      gpu_mem_total_mb — total VRAM (MiB), or None

    Cheap (~5-30 ms) — safe to call on every heartbeat. Failures are
    swallowed; we'd rather report stale numbers than crash the heartbeat
    loop.
    """
    out = {"cpu_percent": 0.0, "gpu_percent": None,
           "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    try:
        import psutil
        # interval=None returns the percent since the LAST call. The
        # heartbeat thread calls this every HEARTBEAT_SEC (15 s), so
        # the value is the average over that window. First call after
        # process start always returns 0.0 — that's expected.
        out["cpu_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            # If multiple GPUs, take GPU 0 (the one CUDA exposes by default).
            parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                out["gpu_percent"]      = round(float(parts[0]), 1)
                out["gpu_mem_used_mb"]  = int(float(parts[1]))
                out["gpu_mem_total_mb"] = int(float(parts[2]))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    except Exception:
        pass
    return out


# ─── Task execution ──────────────────────────────────────────────────────────

def _execute_task(task: dict) -> dict:
    """
    Run one training task. Returns result dict with metrics.
    Dispatches to the appropriate training module based on model_type.
    """
    model_type = task.get("model_type", "")
    symbol     = task.get("symbol", "BTC/USDT")
    timeframe  = task.get("timeframe", "1m")
    config     = task.get("config", {})
    data_path  = task.get("data_path", "")
    output_path = task.get("output_path", str(PROJECT_ROOT / "models"))

    # ── VRAM capacity guard ───────────────────────────────────────────────────
    # TFT needs ≥6 GB free VRAM. Workers below this threshold return a reroute
    # signal so the orchestrator can reassign the task to PC2/PC3.
    if model_type == "tft":
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb < 6.0:
                    logger.warning(
                        "[Worker] Rejecting TFT task — VRAM %.1f GB < 6 GB minimum. "
                        "Orchestrator should reroute to a larger GPU worker.",
                        vram_gb,
                    )
                    return {
                        "error": f"insufficient_vram:{vram_gb:.1f}GB",
                        "status": "failed",
                        "reroute": True,
                    }
        except Exception:
            pass  # no torch → proceed, training will fail gracefully

    logger.info("[Worker] Running task: %s / %s / %s", model_type, symbol, timeframe)

    # 2026-05-10 — synthetic CPU+GPU stress test. Lets an operator
    # verify a worker is reachable, configured correctly, and actually
    # exercising both compute lanes — without touching real model
    # files. Submitted as
    #   {"model_type": "smoke_test", "config": {"duration_s": 300}}
    # Skips the master_trainer dispatch entirely.
    if model_type == "smoke_test":
        return _run_smoke_test(task)

    # Phase 94 — distributed backtest cell. Each task is one
    # (symbol, timeframe) cell; the worker loads its own data via the
    # SMB-mounted project root, builds signals, runs all enabled
    # strategies + meta-filtered variants, and returns summary dicts.
    # See run_distributed_backtest in src/engine/backtester.py.
    if model_type == "backtest_cell":
        return _run_backtest_cell(task)

    # v4 Phase B2 (2026-05-09) — invoke master's actual trainer modules
    # via the SMB-mounted code share, not the worker's simplified RF
    # wrapper. This is what makes the worker a peer-equal node to master:
    # same training logic, same artifact format, same meta JSON shape.
    # Falls back to the legacy generic-RF handlers if the master trainer
    # import fails (e.g. missing dep on the worker venv).
    if config.get("use_master_trainer", True):
        master_result = _invoke_master_trainer(model_type, timeframe, symbol, config)
        if master_result is not None:
            return master_result
        logger.warning("[Worker] master-trainer dispatch failed for %s — falling back to generic RF",
                       model_type)

    # Legacy generic-RF handlers (kept as fallback so a worker-side
    # import bug doesn't take the worker offline).
    handlers = {
        "btc_rf":        _train_random_forest,
        "trend":         _train_sklearn_model,
        "scalping":      _train_sklearn_model,
        "meta_labeler":  _train_sklearn_model,
        "futures_short": _train_sklearn_model,
        "regime":        _train_sklearn_model,
        "tft":           _train_tft,
        "oft":           _train_oft,
        "garch":         _train_garch,
    }
    handler = handlers.get(model_type, _train_sklearn_model)
    return handler(task)


# ─── v4 Phase B2 — master-trainer dispatch ─────────────────────────────────────

# Map of cluster task `model_type` → (master module, function name). The
# master function MUST accept a `timeframe=` kwarg and read its symbol
# universe from the project config; that's the existing contract for
# every trainer in src/engine/. Workers running this dispatch must have
# PYTHONPATH=Z:\ (mounted master share) so `from src.engine...` resolves
# to master's code, not stale worker-local copies.
_MASTER_TRAINER_DISPATCH = {
    "base":          ("src.engine.train_model",          "train_model"),
    "btc_rf":        ("src.engine.train_model",          "train_model"),       # legacy alias
    "trend":         ("src.engine.train_trend_model",    "train_trend_model"),
    "futures":       ("src.engine.train_futures_model",  "train_futures_model"),
    "futures_short": ("src.engine.train_futures_model",  "train_futures_model"),  # legacy alias
    "scalping":      ("src.engine.train_scalping_model", "train_scalping_model"),
    "meta":          ("src.engine.train_meta_labeler",   "train_meta_labeler"),
    "meta_labeler":  ("src.engine.train_meta_labeler",   "train_meta_labeler"),  # legacy alias
    "tft":           ("src.engine.train_tft_model",      "train_tft_model"),
    "regime":        ("src.analysis.regime_classifier",  "train_regime_classifier"),
    "oft":           ("src.training.joint_oft_rl",       "train_oft"),
}


def _invoke_master_trainer(model_type: str, timeframe: str, symbol: str,
                           config: dict) -> dict | None:
    """Import master's trainer module and call it with the task's timeframe.
    Returns a result dict with metrics, or None if the import / call
    failed and the caller should fall back to the legacy handler."""
    spec = _MASTER_TRAINER_DISPATCH.get(model_type)
    if not spec:
        logger.info("[Worker] no master trainer registered for model_type=%r — using legacy handler", model_type)
        return None
    mod_name, fn_name = spec
    import importlib, time as _time, json as _json, os as _os
    t0 = _time.time()
    try:
        mod = importlib.import_module(mod_name)
        fn  = getattr(mod, fn_name)
    except Exception as exc:
        logger.warning("[Worker] master trainer import failed (%s.%s): %s", mod_name, fn_name, exc)
        return None
    # 2026-05-10 — Bug #1 fix. Master trainers use relative paths
    # ('data/raw/BTC_USDT_1h.csv.gz') resolved against the current working
    # directory. On a remote worker, cwd is the worker's local dir, NOT
    # the SMB-mounted Z: share, so every load failed with "No training
    # data found for ALL/<tf>". Chdir to PROJECT_ROOT (which is Z:\ on
    # the worker via SMB) so relative paths resolve correctly. Restore
    # cwd after so we don't pollute the worker process state.
    saved_cwd = _os.getcwd()
    try:
        _os.chdir(str(PROJECT_ROOT))
    except Exception as exc:
        logger.warning("[Worker] chdir(%s) failed: %s — trainer may fail to find data",
                       PROJECT_ROOT, exc)
    # Some trainers (oft, regime) don't accept timeframe — call accordingly.
    try:
        if model_type in ("oft",):
            # train_oft(symbol, timeframe, ...) — special signature
            result = fn(symbol, timeframe)
        elif model_type in ("regime",):
            # train_regime_classifier() — no kwargs
            result = fn()
        else:
            result = fn(timeframe=timeframe)
    except Exception as exc:
        logger.exception("[Worker] master trainer raised: %s.%s(timeframe=%r): %s",
                         mod_name, fn_name, timeframe, exc)
        return {
            "status":  "failed",
            "error":   f"master_trainer_exception: {type(exc).__name__}: {exc}",
            "model":   model_type,
            "timeframe": timeframe,
            "symbol":  symbol,
            "duration_s": round(_time.time() - t0, 2),
            "trainer_path": f"{mod_name}.{fn_name}",
        }
    finally:
        try:
            _os.chdir(saved_cwd)
        except Exception:
            pass
    duration = _time.time() - t0
    # Read the meta JSON the trainer just wrote to extract metrics —
    # standard contract: every trainer writes models/<key>_<tf>_meta.json
    # (or models/<key>_meta.json for canonical TF + special models like
    # regime, meta_labeler).
    metrics = {}
    try:
        from src.utils.model_paths import artifact_paths, KEYS as _MODEL_KEYS
        # Map task model_type to a canonical key the model_paths helper
        # knows about (futures_short → futures, btc_rf → base, etc.).
        canon = {"futures_short": "futures", "btc_rf": "base",
                 "meta_labeler": "meta"}.get(model_type, model_type)
        if canon in _MODEL_KEYS:
            paths = artifact_paths(canon, timeframe)
            meta_path = paths.get("meta")
            if meta_path and meta_path.exists():
                metrics = _json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        # Metrics-read failure isn't fatal — trainer succeeded, just no metric surface.
        pass
    return {
        "status":      "done",
        "model":       model_type,
        "timeframe":   timeframe,
        "symbol":      symbol,
        "duration_s":  round(duration, 2),
        "trainer_path": f"{mod_name}.{fn_name}",
        "via":         "master_trainer",
        "metrics":     {
            "accuracy":              metrics.get("accuracy"),
            "walk_forward_mean_acc": metrics.get("walk_forward_mean_acc"),
            "auc_roc":               metrics.get("auc_roc"),
            "n_samples":             metrics.get("n_samples"),
        } if metrics else {},
    }


# ─── 2026-05-10 — Synthetic CPU+GPU stress test ─────────────────────────────

def _run_smoke_test(task: dict) -> dict:
    """Burn CPU and GPU for a configurable duration so an operator can
    verify a worker node is healthy and visible in compute monitors.

    Task spec:
      {"model_type": "smoke_test",
       "config": {"duration_s": 300}}        # default 60 s, capped at 1800

    What it does:
      - CPU lane: tight numpy matmul loop on ~1024×1024 random arrays
        across (cpu_count - 1) threads, sized to keep ~70-90% CPU
        without thrashing.
      - GPU lane: if torch+CUDA available, runs continuous tensor
        matmuls on small (~512×512) cuda tensors with periodic
        torch.cuda.synchronize() so nvidia-smi sees activity. VRAM
        footprint stays ≤ ~200 MB so it fits even on a 6 GB card.
      - Logs every 30 s with elapsed/remaining/CPU%/GPU%.

    Returns standard task result dict with status='done' on completion.
    Stops cleanly when duration elapses; no real model artifacts touched.
    """
    import time as _time, math as _math, threading as _threading
    cfg = task.get("config", {}) or {}
    duration_s = max(5, min(1800, int(cfg.get("duration_s", 60))))
    t_start = _time.time()
    t_end   = t_start + duration_s

    logger.info("[Worker] SMOKE_TEST start: duration=%ds — CPU lane + GPU lane",
                duration_s)

    # ── CPU lane ────────────────────────────────────────────────────────
    cpu_done = _threading.Event()
    cpu_iters = [0]
    def _cpu_loop():
        try:
            import numpy as _np
            # Size chosen so a single matmul takes ~30-60 ms on modern
            # CPUs — small enough to keep iteration rate high (granular
            # cancellation) but large enough that TaskManager registers
            # the load.
            A = _np.random.rand(1024, 1024).astype(_np.float32)
            B = _np.random.rand(1024, 1024).astype(_np.float32)
            while _time.time() < t_end:
                _np.dot(A, B)
                cpu_iters[0] += 1
        finally:
            cpu_done.set()

    import os as _os
    n_cpu_threads = max(1, (_os.cpu_count() or 4) - 1)
    cpu_threads = [_threading.Thread(target=_cpu_loop, daemon=True,
                                      name=f"smoke-cpu-{i}")
                   for i in range(n_cpu_threads)]
    for t in cpu_threads:
        t.start()

    # ── GPU lane ────────────────────────────────────────────────────────
    gpu_iters = [0]
    gpu_available = False
    gpu_thread = None
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            gpu_available = True
            def _gpu_loop():
                try:
                    a = _torch.randn(512, 512, device='cuda')
                    b = _torch.randn(512, 512, device='cuda')
                    while _time.time() < t_end:
                        c = a @ b
                        a = c + 0.001
                        # Force sync every 32 iterations so utilisation
                        # registers in nvidia-smi instead of being
                        # masked by async kernel queueing.
                        if gpu_iters[0] % 32 == 0:
                            _torch.cuda.synchronize()
                        gpu_iters[0] += 1
                except Exception as exc:
                    logger.warning("[Worker] smoke_test GPU lane error: %s", exc)
            gpu_thread = _threading.Thread(target=_gpu_loop, daemon=True,
                                            name="smoke-gpu")
            gpu_thread.start()
    except Exception as exc:
        logger.info("[Worker] smoke_test: torch/CUDA unavailable (%s) — CPU-only", exc)

    # ── Heartbeat log every 30 s ────────────────────────────────────────
    while _time.time() < t_end:
        _time.sleep(min(30.0, t_end - _time.time()))
        elapsed = int(_time.time() - t_start)
        remain  = max(0, int(t_end - _time.time()))
        logger.info("[Worker] SMOKE_TEST tick: elapsed=%ds remain=%ds "
                    "cpu_iters=%d gpu_iters=%d gpu=%s",
                    elapsed, remain, cpu_iters[0], gpu_iters[0],
                    'on' if gpu_available else 'off')

    # ── Drain ───────────────────────────────────────────────────────────
    for t in cpu_threads:
        t.join(timeout=2.0)
    if gpu_thread is not None:
        gpu_thread.join(timeout=2.0)
    duration = _time.time() - t_start
    logger.info("[Worker] SMOKE_TEST done: %.1fs cpu_iters=%d gpu_iters=%d",
                duration, cpu_iters[0], gpu_iters[0])

    return {
        "status":     "done",
        "model":      "smoke_test",
        "timeframe":  task.get("timeframe", "-"),
        "symbol":     task.get("symbol", "-"),
        "duration_s": round(duration, 2),
        "via":        "smoke_test",
        "metrics": {
            "cpu_iters":        cpu_iters[0],
            "gpu_iters":        gpu_iters[0],
            "gpu_available":    gpu_available,
            "n_cpu_threads":    n_cpu_threads,
            "requested_duration_s": duration_s,
        },
    }


# ─── Phase 94 — distributed backtest cell handler ───────────────────────────
#
# What this is: the worker-side counterpart to
# `run_distributed_backtest` in src/engine/backtester.py. The
# orchestrator dispatches one task per (symbol, timeframe) cell with
# model_type='backtest_cell'; this handler imports master's actual
# backtest code (via the SMB-mounted Z:\) and returns summary metrics
# for every enabled strategy on that cell. Trades and equity curves
# stay on the worker — only summary dicts cross the wire, so the
# task_update payload is small (a few KB per cell vs. tens of MB if we
# shipped raw BacktestResult objects).

def _run_backtest_cell(task: dict) -> dict:
    """Run all enabled backtest strategies for one (symbol, timeframe)
    cell. Task spec:
      {"model_type": "backtest_cell",
       "symbol":     "BTC_USDT",
       "timeframe":  "1h",
       "config": {"initial_capital": 10000.0,
                  "fee_preset":      "futures",
                  "models":          ["base", "trend", ...] | None}}

    Returns a result dict the orchestrator merges into the task table.
    metrics.strategy_results is the list of per-strategy summary
    dicts that master aggregates into the comparison frame.
    """
    import time as _time, os as _os
    cfg       = task.get("config", {}) or {}
    symbol    = task.get("symbol", "BTC_USDT")
    timeframe = task.get("timeframe", "1h")

    # Same chdir trick as _invoke_master_trainer: master's backtester
    # uses relative paths ('data/raw/...') resolved against cwd. On a
    # remote worker, cwd is the worker's local dir not Z:\, so without
    # this every cell would fail "No training data found". chdir to
    # PROJECT_ROOT (which is Z:\ on the worker via SMB) so relative
    # paths resolve to the master's data tree.
    saved_cwd = _os.getcwd()
    try:
        _os.chdir(str(PROJECT_ROOT))
    except Exception:
        pass

    t0 = _time.time()
    try:
        from src.engine.backtester import run_one_backtest_cell_summaries
    except Exception as exc:
        try: _os.chdir(saved_cwd)
        except Exception: pass
        return {"status": "failed", "model": "backtest_cell",
                "symbol": symbol, "timeframe": timeframe,
                "error": f"backtester import failed: {type(exc).__name__}: {exc}"}

    models_cfg = cfg.get("models")
    models_tup = tuple(models_cfg) if models_cfg else None

    try:
        rows = run_one_backtest_cell_summaries(
            symbol, timeframe,
            initial_capital=float(cfg.get("initial_capital", 10_000.0)),
            fee_preset=str(cfg.get("fee_preset", "futures")),
            models=models_tup,
        )
    except Exception as exc:
        logger.exception("[Worker] backtest_cell %s/%s raised", symbol, timeframe)
        try: _os.chdir(saved_cwd)
        except Exception: pass
        return {"status": "failed", "model": "backtest_cell",
                "symbol": symbol, "timeframe": timeframe,
                "duration_s": round(_time.time() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try: _os.chdir(saved_cwd)
        except Exception: pass

    duration = _time.time() - t0
    logger.info("[Worker] backtest_cell %s/%s done in %.1fs (%d strategies)",
                symbol, timeframe, duration, len(rows))
    return {
        "status":     "done",
        "model":      "backtest_cell",
        "symbol":     symbol,
        "timeframe":  timeframe,
        "duration_s": round(duration, 2),
        "via":        "backtest_cell",
        "metrics":    {
            "strategy_results": rows,
            "n_strategies":     len(rows),
        },
    }


def _load_data(data_path: str, symbol: str, timeframe: str):
    """Load training data from path or fall back to project data directory."""
    import pandas as pd
    import gzip

    # Try explicit data_path first (UNC path from master)
    if data_path and Path(data_path).exists():
        path = Path(data_path)
    else:
        # Fall back to local data directory
        safe = symbol.replace("/", "_")
        candidates = [
            PROJECT_ROOT / "data" / "raw" / f"{safe}_{timeframe}.csv.gz",
            PROJECT_ROOT / "data" / "raw" / "historical" / f"{safe}_spot_1s.csv.gz",
        ]
        path = next((p for p in candidates if p.exists()), None)

    if path is None or not path.exists():
        raise FileNotFoundError(f"No training data found for {symbol}/{timeframe}")

    logger.info("[Worker] Loading data: %s (%.0f MB)", path.name, path.stat().st_size / 1e6)
    df = pd.read_csv(path, compression="gzip", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _train_sklearn_model(task: dict) -> dict:
    """Generic sklearn model training (RF, gradient boosting)."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    symbol    = task.get("symbol", "BTC/USDT")
    timeframe = task.get("timeframe", "1m")
    config    = task.get("config", {})
    model_type = task.get("model_type", "trend")
    output_path = Path(task.get("output_path", str(PROJECT_ROOT / "models")))
    output_path.mkdir(parents=True, exist_ok=True)

    df = _load_data(task.get("data_path", ""), symbol, timeframe)

    # Build features from OHLCV
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    close = df["close"]
    df["ret_1"]  = close.pct_change(1)
    df["ret_5"]  = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)
    df["vol_10"] = close.pct_change().rolling(10).std()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]
    df["target"] = (close.shift(-5) > close).astype(int)
    df = df.dropna()

    feat_cols = ["ret_1", "ret_5", "ret_20", "vol_10", "sma_ratio"]
    X = df[feat_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    n_estimators = config.get("n_estimators", 100)
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    model_file = output_path / f"{model_type}_{symbol.replace('/','_')}_worker.joblib"
    joblib.dump(model, model_file)

    return {
        "accuracy":   round(float(acc), 4),
        "n_samples":  len(X_train),
        "model_file": str(model_file),
        "features":   feat_cols,
    }


def _train_random_forest(task: dict) -> dict:
    return _train_sklearn_model(task)


def _train_tft(task: dict) -> dict:
    """TFT model training — uses darts if available."""
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.engine.train_tft_model import train_tft
        symbol    = task.get("symbol", "BTC/USDT")
        timeframe = task.get("timeframe", "1m")
        result = train_tft(symbol=symbol, timeframe=timeframe)
        return result or {"status": "done"}
    except Exception as exc:
        return {"error": str(exc), "status": "failed"}


def _train_oft(task: dict) -> dict:
    """OFT (Order Flow Transformer) — Phase 2 microstructure model.

    Defers to `src.training.joint_oft_rl.train_oft` which writes the
    checkpoint to models/oft_model.pt. SAC stage is skipped because the
    distributed cluster only handles the supervised half today."""
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.training.joint_oft_rl import train_oft as _tof
        symbol    = task.get("symbol", "BTC/USDT")
        timeframe = task.get("timeframe", "1m")
        cfg       = task.get("config", {}) or {}
        epochs    = int(cfg.get("epochs", 5))
        result = _tof(symbol=symbol, timeframe=timeframe, n_epochs=epochs)
        return {**(result or {}), "status": "done"}
    except Exception as exc:
        return {"error": str(exc), "status": "failed"}


def _train_garch(task: dict) -> dict:
    """GARCH volatility model."""
    import numpy as np
    try:
        from arch import arch_model
        df = _load_data(task.get("data_path", ""), task.get("symbol", "BTC/USDT"), task.get("timeframe", "1m"))
        returns = df["close"].pct_change().dropna() * 100
        am = arch_model(returns.tail(5000), vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        return {"aic": round(float(res.aic), 2), "bic": round(float(res.bic), 2), "status": "done"}
    except Exception as exc:
        return {"error": str(exc), "status": "failed"}


# ─── Worker HTTP server ───────────────────────────────────────────────────────

class TrainingWorker:

    def __init__(self, master_url: str, node_id: str, name: str, port: int = WORKER_PORT,
                 lane: str = "any"):
        self.master_url = master_url.rstrip("/")
        self.node_id    = node_id
        self.name       = name
        self.port       = port
        # 2026-05-10 — dual-lane support. lane = "cpu" | "gpu" | "any".
        # "any" (default) = legacy behaviour, accepts any task. "cpu" only
        # accepts cpu-lane tasks; "gpu" only accepts gpu/exclusive tasks.
        # The orchestrator filters dispatch by lane match. This lets a
        # single PC run two worker processes — one per lane — so CPU and
        # GPU work happen concurrently on the same machine.
        self.lane       = lane if lane in ("cpu", "gpu", "any") else "any"
        self.hw         = _detect_hardware()
        self._running   = False
        self._current_task: dict | None = None
        self._lock      = threading.Lock()
        # Phase 93 — uptime tracking for /system_info and remote-restart
        # confirmation. master_agent uses this to verify a /restart POST
        # actually rolled the worker over (uptime drops to <heartbeat).
        self._start_time = time.time()
        self._app       = self._build_app()

    def _build_app(self):
        from flask import Flask, jsonify, request as freq
        app = Flask(f"worker-{self.node_id}")
        app.logger.setLevel(logging.WARNING)

        @app.route("/health")
        def health():
            return jsonify({
                "node_id": self.node_id,
                "name": self.name,
                "status": "busy" if self._current_task else "idle",
                "hw": self.hw,
                "transport": self._transport_info(),
                # Phase 93 — live CPU+GPU sample on demand. Cheap enough
                # (~5-30 ms) to include in every /health response.
                "live_load": _sample_live_load(),
                "uptime_s":  int(time.time() - self._start_time),
            })

        @app.route("/task", methods=["POST"])
        def run_task():
            task = freq.get_json(force=True) or {}
            if self._current_task:
                return jsonify({"error": "busy"}), 409
            threading.Thread(target=self._run_task, args=(task,), daemon=True).start()
            return jsonify({"ok": True, "task_id": task.get("task_id")})

        @app.route("/cancel", methods=["POST"])
        def cancel():
            self._current_task = None
            return jsonify({"ok": True})

        # Phase 93 — remote process control. master_agent POSTs here to
        # remediate a remote zombie worker (host != LOCAL_HOSTNAME).
        # Pre-Phase-93 the only remediation was an alert + manual SSH —
        # now the supervisor can self-heal even when the worker is on
        # Ivan's PC (or any future remote node).
        #
        # Confirm gate: caller MUST send {"confirm": true} in the body.
        # Reason: 2026-05-10 a probe POST with {"dry_run": true} (a flag
        # this endpoint does NOT honour) re-execed Ivan's GPU worker by
        # accident. The gate keeps `/restart` reachable on the LAN
        # without auth but blocks single-line copy-paste mistakes.
        # master_agent and the dashboard proxy both send the flag.
        #
        # Implementation: respond OK first, then re-exec the same Python
        # process in a background thread after a 1-second delay so the
        # HTTP response flushes before the listening socket dies. On
        # failure we fall back to os._exit(1); a process supervisor
        # (master_agent / Windows service / systemd) is responsible for
        # restarting us — same recovery path as crash-on-startup.
        @app.route("/restart", methods=["POST"])
        def restart():
            body = freq.get_json(force=True, silent=True) or {}
            if body.get("confirm") is not True:
                return jsonify({
                    "ok": False,
                    "error": "confirm flag required",
                    "hint": 'POST body must include {"confirm": true}',
                }), 400
            def _delayed_exec():
                time.sleep(1.0)
                logger.warning("[Worker] /restart received — re-executing self")
                try:
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception as exc:
                    logger.error("[Worker] /restart os.execv failed: %s — exiting", exc)
                    os._exit(1)
            threading.Thread(target=_delayed_exec, daemon=True,
                             name="worker-restart").start()
            return jsonify({"ok": True, "action": "restart_in_1s",
                            "node_id": self.node_id, "name": self.name})

        # Phase 93 — full diagnostic dump. Hit this when investigating
        # "why is this worker behaving weirdly" without having to read
        # logs on the remote machine.
        @app.route("/system_info")
        def system_info():
            return jsonify({
                "node_id":      self.node_id,
                "name":         self.name,
                "lane":         self.lane,
                "port":         self.port,
                "hw":           self.hw,
                "live_load":    _sample_live_load(),
                "transport":    self._transport_info(),
                "current_task": (self._current_task or {}).get("task_id", ""),
                "uptime_s":     int(time.time() - self._start_time),
                "python":       sys.version.split()[0],
                "platform":     platform.platform(),
                "pid":          os.getpid(),
            })

        return app

    def _run_task(self, task: dict) -> None:
        task_id = task.get("task_id", "?")
        with self._lock:
            self._current_task = task

        self._notify_master("running", task_id)
        # 2026-05-11 — defensive task heartbeat. Some trainers (Darts/
        # Lightning's TFT, in particular) report progress only via tqdm to
        # stdout; they never call back to the orchestrator. The cluster
        # watchdog uses `last_update_at` to decide "is this task stale",
        # so a healthy multi-hour neural training looks identical to a
        # zombie crash. This thread POSTs status="heartbeat" every
        # TASK_HEARTBEAT_S so `last_update_at` stays fresh for the full
        # task duration. The orchestrator's update_task() treats
        # "heartbeat" as a stale-window refresh only (no state change).
        heartbeat_stop = threading.Event()
        def _task_heartbeat():
            while not heartbeat_stop.wait(TASK_HEARTBEAT_S):
                self._notify_master("heartbeat", task_id)
        hb_thread = threading.Thread(
            target=_task_heartbeat, daemon=True,
            name=f"task-heartbeat-{task_id[:8]}",
        )
        hb_thread.start()
        try:
            result = _execute_task(task)
            # 2026-05-11 fix — _execute_task / _invoke_master_trainer catches
            # trainer exceptions and returns {"status": "failed", "error": ...}
            # in the result dict. Pre-fix this branch reported "done" to the
            # orchestrator regardless, so silent training failures were
            # invisible (e.g. task e5bba811-21c btc_rf @ 5m failed with
            # XGBClassifierWrapper sklearn-type error but cluster said done,
            # leaving operator unable to tell training succeeded vs silently
            # failed). Now honor the worker-side status from the result dict.
            inner_status = (result.get('status', 'done')
                             if isinstance(result, dict) else 'done')
            if inner_status == 'failed':
                self._notify_master(
                    "failed", task_id, result=result,
                    error=(result.get('error', '')
                           if isinstance(result, dict) else ''),
                )
                logger.error("[Worker] Task %s FAILED (trainer error): %s",
                              task_id, result.get('error') if isinstance(result, dict) else result)
            else:
                self._notify_master("done", task_id, result=result)
                logger.info("[Worker] Task %s DONE: %s", task_id, result)
        except Exception as exc:
            self._notify_master("failed", task_id, error=str(exc))
            logger.error("[Worker] Task %s FAILED (exception): %s", task_id, exc)
        finally:
            heartbeat_stop.set()  # Signal heartbeat to exit; daemon=True will GC.
            with self._lock:
                self._current_task = None

    def _transport_info(self) -> dict:
        """Phase 0: surface ZMQ data-bus availability for dashboard visibility.

        Importing data_bus does not bind sockets — sockets open lazily on
        first publish/subscribe. Workers will use this in Phase 3 (joint
        OFT+RL training) to pull mini-batches from the master.
        """
        try:
            from src.transport.data_bus import get_data_bus
            return {"zmq_available": True, **get_data_bus().stats()}
        except Exception as exc:
            return {"zmq_available": False, "error": str(exc)}

    def _notify_master(self, status: str, task_id: str, result: dict | None = None, error: str = "") -> None:
        import requests as req
        try:
            req.post(
                f"{self.master_url}/api/cluster/task_update",
                json={"task_id": task_id, "node_id": self.node_id,
                      "status": status, "result": result or {}, "error": error},
                timeout=10,
            )
        except Exception as exc:
            logger.debug("[Worker] Notify master failed: %s", exc)

    def _heartbeat_loop(self) -> None:
        import requests as req
        while self._running:
            try:
                # Phase 93 — sample live load once per heartbeat and ship
                # it inline. Adds ~30 ms to each beat (nvidia-smi spawn);
                # acceptable at HEARTBEAT_SEC=15. Numbers go straight
                # through register_worker into the cluster's worker dict
                # and on to the dashboard's Live Load column.
                live = _sample_live_load()
                req.post(
                    f"{self.master_url}/api/cluster/register",
                    json={
                        "node_id":       self.node_id,
                        "name":          self.name,
                        "hostname":      self.hw["hostname"],
                        "ip":            self.hw["ip"],
                        "port":          self.port,
                        "gpu_name":      self.hw["gpu_name"],
                        "gpu_vram_gb":   self.hw["gpu_vram_gb"],
                        "cpu_cores":     self.hw["cpu_cores"],
                        "ram_gb":        self.hw["ram_gb"],
                        "cuda_available": self.hw["cuda_available"],
                        "lane":          self.lane,
                        "status":        "busy" if self._current_task else "idle",
                        "current_task":  (self._current_task or {}).get("task_id", ""),
                        "last_seen":     datetime.now(timezone.utc).isoformat(),
                        # Phase 93 live load
                        "cpu_percent":      live["cpu_percent"],
                        "gpu_percent":      live["gpu_percent"],
                        "gpu_mem_used_mb":  live["gpu_mem_used_mb"],
                        "gpu_mem_total_mb": live["gpu_mem_total_mb"],
                        "uptime_s":         int(time.time() - self._start_time),
                    },
                    timeout=5,
                )
            except Exception:
                pass
            time.sleep(HEARTBEAT_SEC)

    def start(self) -> None:
        _ensure_deps()
        self._running = True
        hw = self.hw
        logger.info("=" * 60)
        logger.info("Training Worker  —  %s  [%s]", self.name, self.node_id)
        logger.info("GPU: %s  (CUDA: %s)  VRAM: %.1f GB", hw["gpu_name"], hw["cuda_available"], hw["gpu_vram_gb"])
        logger.info("CPU: %d cores  RAM: %.1f GB", hw["cpu_cores"], hw["ram_gb"])
        logger.info("Master: %s", self.master_url)
        logger.info("Listening on port %d", self.port)
        logger.info("=" * 60)

        threading.Thread(target=self._heartbeat_loop, daemon=True, name="worker-hb").start()
        self._app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="AI Trading — Training Worker Node")
    parser.add_argument("--master", required=True, metavar="URL",
                        help="Master orchestrator URL, e.g. http://192.168.1.100:7700")
    parser.add_argument("--name",   default=socket.gethostname(), help="Human-readable node name")
    parser.add_argument("--port",   type=int, default=WORKER_PORT, help="Local HTTP port (default 7701)")
    parser.add_argument("--id",     default=str(uuid.uuid4())[:8],  help="Node ID (auto-generated)")
    parser.add_argument("--lane",   default="any", choices=("cpu", "gpu", "any"),
                        help="Which task lane this worker accepts. 'cpu' = only "
                             "CPU-resource models (base/trend/futures/scalping/meta/regime). "
                             "'gpu' = only gpu/exclusive (tft/oft). 'any' = legacy/default.")
    args = parser.parse_args()

    # 2026-05-11 fix — CPU-lane workers must NOT touch any GPU. Operator
    # screenshot showed Intel iGPU at 67% from a --lane cpu worker because
    # PyTorch/sklearn import paths probe ALL visible GPU adapters at boot,
    # allocating context on the integrated GPU even though all training
    # work would run on CPU. Hide every CUDA device from a CPU-lane worker
    # so the imports can't bind any adapter. Must run BEFORE TrainingWorker
    # imports the heavy ML stack — argparse import is fine here.
    if args.lane == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Some libs (e.g. older torch builds) also probe MPS/ROCm; clear
        # those too for completeness. No-op when those env vars aren't read.
        os.environ['HIP_VISIBLE_DEVICES']  = ''
        logging.getLogger(__name__).info(
            "[worker] lane=cpu — hiding all GPU adapters via "
            "CUDA_VISIBLE_DEVICES='' / HIP_VISIBLE_DEVICES=''")

    worker = TrainingWorker(
        master_url=args.master,
        node_id=args.id,
        name=args.name,
        port=args.port,
        lane=args.lane,
    )
    worker.start()


if __name__ == "__main__":
    main()
