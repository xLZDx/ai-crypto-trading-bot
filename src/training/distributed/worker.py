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
HEARTBEAT_SEC = 15          # heartbeat interval


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
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


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

    logger.info("[Worker] Running task: %s / %s / %s", model_type, symbol, timeframe)

    # Map model_type → training function
    handlers = {
        "btc_rf":        _train_random_forest,
        "trend":         _train_sklearn_model,
        "scalping":      _train_sklearn_model,
        "meta_labeler":  _train_sklearn_model,
        "futures_short": _train_sklearn_model,
        "regime":        _train_sklearn_model,
        "tft":           _train_tft,
        "garch":         _train_garch,
    }
    handler = handlers.get(model_type, _train_sklearn_model)
    return handler(task)


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

    def __init__(self, master_url: str, node_id: str, name: str, port: int = WORKER_PORT):
        self.master_url = master_url.rstrip("/")
        self.node_id    = node_id
        self.name       = name
        self.port       = port
        self.hw         = _detect_hardware()
        self._running   = False
        self._current_task: dict | None = None
        self._lock      = threading.Lock()
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

        return app

    def _run_task(self, task: dict) -> None:
        task_id = task.get("task_id", "?")
        with self._lock:
            self._current_task = task

        self._notify_master("running", task_id)
        try:
            result = _execute_task(task)
            self._notify_master("done", task_id, result=result)
            logger.info("[Worker] Task %s DONE: %s", task_id, result)
        except Exception as exc:
            self._notify_master("failed", task_id, error=str(exc))
            logger.error("[Worker] Task %s FAILED: %s", task_id, exc)
        finally:
            with self._lock:
                self._current_task = None

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
                        "status":        "busy" if self._current_task else "idle",
                        "current_task":  (self._current_task or {}).get("task_id", ""),
                        "last_seen":     datetime.now(timezone.utc).isoformat(),
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
    args = parser.parse_args()

    worker = TrainingWorker(
        master_url=args.master,
        node_id=args.id,
        name=args.name,
        port=args.port,
    )
    worker.start()


if __name__ == "__main__":
    main()
