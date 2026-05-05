"""
Training Orchestrator — runs on the master laptop.

Manages a cluster of worker nodes (other laptops running worker.py).
Assigns training tasks based on worker capabilities (GPU first, then CPU).
Exposes a REST API consumed by the dashboard monitor tab.

Start:
    python -m src.training.distributed.orchestrator
    python -m src.training.distributed.orchestrator --port 7700

Dashboard API (registered in app.py):
    GET  /api/cluster/status          — cluster overview
    GET  /api/cluster/workers         — list all workers
    POST /api/cluster/submit          — submit a training task
    POST /api/cluster/register        — worker heartbeat/register
    POST /api/cluster/task_update     — worker reports task result
    DELETE /api/cluster/task/<id>     — cancel a task
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ORCH_PORT       = 7700
WORKER_TIMEOUT  = 60    # mark worker offline after N seconds without heartbeat
MAX_TASK_RETRIES = 2


class Orchestrator:
    """
    In-process orchestrator — can be embedded in the dashboard Flask app
    or run as a standalone process.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._workers: dict[str, dict] = {}      # node_id → WorkerInfo dict
        self._tasks:   dict[str, dict] = {}       # task_id → TrainingTask dict
        self._queue:   list[str]       = []       # task_ids in order
        self._running  = False
        self._schedule_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._schedule_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="orch-scheduler"
        )
        self._schedule_thread.start()
        logger.info("[Orch] Orchestrator started")

    def stop(self) -> None:
        self._running = False

    # ── Worker registration ───────────────────────────────────────────────────

    def register_worker(self, info: dict) -> None:
        node_id = info.get("node_id", "")
        if not node_id:
            return
        with self._lock:
            prev = self._workers.get(node_id, {})
            info["last_seen"] = datetime.now(timezone.utc).isoformat()
            # Don't overwrite status if we just assigned it a task
            if prev.get("status") == "busy" and info.get("status") == "idle" and prev.get("current_task"):
                info["status"] = "busy"
            self._workers[node_id] = {**prev, **info}
        logger.debug("[Orch] Worker registered: %s (%s)", info.get("name", node_id), info.get("ip"))

    def list_workers(self) -> list[dict]:
        now = time.time()
        result = []
        with self._lock:
            for w in self._workers.values():
                w = dict(w)
                # Calculate seconds since last seen
                try:
                    ls = datetime.fromisoformat(w.get("last_seen", "").replace("Z", "+00:00"))
                    age = now - ls.timestamp()
                    w["online"] = age < WORKER_TIMEOUT
                    w["last_seen_ago"] = int(age)
                except Exception:
                    w["online"] = False
                    w["last_seen_ago"] = 9999
                result.append(w)
        return result

    # ── Task submission ───────────────────────────────────────────────────────

    def submit_task(self, task_spec: dict) -> str:
        """Submit a training task. Returns task_id."""
        task_id = str(uuid.uuid4())[:12]
        task = {
            "task_id":     task_id,
            "model_type":  task_spec.get("model_type", "btc_rf"),
            "symbol":      task_spec.get("symbol", "BTC/USDT"),
            "timeframe":   task_spec.get("timeframe", "1m"),
            "config":      task_spec.get("config", {}),
            "data_path":   task_spec.get("data_path", ""),
            "output_path": task_spec.get("output_path", str(PROJECT_ROOT / "models")),
            "status":      "pending",
            "assigned_to": "",
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "started_at":  "",
            "finished_at": "",
            "result":      {},
            "error":       "",
            "retries":     0,
        }
        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
        logger.info("[Orch] Task submitted: %s / %s / %s", task_id, task["model_type"], task["symbol"])
        return task_id

    def update_task(self, task_id: str, status: str, node_id: str = "",
                    result: dict | None = None, error: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task["status"] = status
            if node_id:
                task["assigned_to"] = node_id
            if result:
                task["result"] = result
            if error:
                task["error"] = error
            if status == "running":
                task["started_at"] = datetime.now(timezone.utc).isoformat()
                # Update worker status
                if node_id and node_id in self._workers:
                    self._workers[node_id]["status"] = "busy"
                    self._workers[node_id]["current_task"] = task_id
            elif status in ("done", "failed", "cancelled"):
                task["finished_at"] = datetime.now(timezone.utc).isoformat()
                # Free worker
                if node_id and node_id in self._workers:
                    self._workers[node_id]["status"] = "idle"
                    self._workers[node_id]["current_task"] = ""
                    if status == "done":
                        self._workers[node_id]["tasks_done"] = self._workers[node_id].get("tasks_done", 0) + 1
                    else:
                        self._workers[node_id]["tasks_failed"] = self._workers[node_id].get("tasks_failed", 0) + 1
                # Retry on failure
                if status == "failed" and task.get("retries", 0) < MAX_TASK_RETRIES:
                    task["retries"] = task.get("retries", 0) + 1
                    task["status"] = "pending"
                    task["assigned_to"] = ""
                    if task_id not in self._queue:
                        self._queue.append(task_id)
                    logger.warning("[Orch] Task %s failed — retry %d/%d", task_id, task["retries"], MAX_TASK_RETRIES)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task["status"] == "pending":
                task["status"] = "cancelled"
                if task_id in self._queue:
                    self._queue.remove(task_id)
                return True
        return False

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            return dict(self._tasks[task_id]) if task_id in self._tasks else None

    def list_tasks(self, limit: int = 50) -> list[dict]:
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks[:limit]

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while self._running:
            try:
                self._dispatch_pending()
            except Exception as exc:
                logger.debug("[Orch] Scheduler error: %s", exc)
            time.sleep(5)

    def _dispatch_pending(self) -> None:
        with self._lock:
            if not self._queue:
                return
            # Find idle workers (GPU workers get tasks first)
            idle = [
                w for w in self._workers.values()
                if w.get("status") == "idle"
                and w.get("online", True)
                and w.get("last_seen_ago", 0) < WORKER_TIMEOUT
            ]
            if not idle:
                return
            # Sort: GPU workers first, then by VRAM descending
            idle.sort(key=lambda w: (-int(w.get("cuda_available", False)), -w.get("gpu_vram_gb", 0)))
            pending = [tid for tid in self._queue if self._tasks.get(tid, {}).get("status") == "pending"]
            for task_id, worker in zip(pending, idle):
                task = self._tasks[task_id]
                task["status"]      = "running"
                task["assigned_to"] = worker["node_id"]
                task["started_at"]  = datetime.now(timezone.utc).isoformat()
                worker["status"]        = "busy"
                worker["current_task"]  = task_id
                if task_id in self._queue:
                    self._queue.remove(task_id)
                # Dispatch in background (don't hold lock during HTTP call)
                threading.Thread(
                    target=self._send_task_to_worker,
                    args=(worker, dict(task)),
                    daemon=True,
                ).start()

    def _send_task_to_worker(self, worker: dict, task: dict) -> None:
        import requests
        ip, port, node_id = worker["ip"], worker["port"], worker["node_id"]
        try:
            r = requests.post(f"http://{ip}:{port}/task", json=task, timeout=15)
            if r.status_code == 200:
                logger.info("[Orch] Task %s → %s (%s:%s)", task["task_id"], worker.get("name", node_id), ip, port)
            else:
                logger.warning("[Orch] Worker %s rejected task: %s", node_id, r.text[:200])
                self.update_task(task["task_id"], "failed", node_id, error=f"Worker rejected: {r.status_code}")
        except Exception as exc:
            logger.warning("[Orch] Cannot reach worker %s: %s", node_id, exc)
            self.update_task(task["task_id"], "pending", node_id)  # re-queue

    # ── Status summary ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        workers = self.list_workers()
        tasks   = self.list_tasks(100)
        return {
            "workers_total":  len(workers),
            "workers_online": sum(1 for w in workers if w.get("online")),
            "workers_idle":   sum(1 for w in workers if w.get("status") == "idle" and w.get("online")),
            "workers_busy":   sum(1 for w in workers if w.get("status") == "busy"),
            "tasks_pending":  sum(1 for t in tasks if t["status"] == "pending"),
            "tasks_running":  sum(1 for t in tasks if t["status"] == "running"),
            "tasks_done":     sum(1 for t in tasks if t["status"] == "done"),
            "tasks_failed":   sum(1 for t in tasks if t["status"] == "failed"),
            "workers":        workers,
            "recent_tasks":   tasks[:20],
        }

    # ── Batch job helpers ─────────────────────────────────────────────────────

    def submit_full_training_run(self, symbols: list[str] | None = None) -> list[str]:
        """Submit training tasks for all models across all symbols."""
        if symbols is None:
            watchlist_file = PROJECT_ROOT / "data" / "watchlist.json"
            symbols = json.loads(watchlist_file.read_text()) if watchlist_file.exists() else ["BTC/USDT"]

        model_configs = [
            {"model_type": "btc_rf",        "timeframe": "1m", "config": {"n_estimators": 200}},
            {"model_type": "trend",          "timeframe": "1h", "config": {}},
            {"model_type": "scalping",       "timeframe": "1m", "config": {}},
            {"model_type": "meta_labeler",   "timeframe": "1m", "config": {}},
            {"model_type": "futures_short",  "timeframe": "1m", "config": {}},
            {"model_type": "regime",         "timeframe": "1h", "config": {}},
            # OFT (Order Flow Transformer) — single-symbol single-machine in
            # current implementation, but listed so the cluster scheduler
            # picks it up once joint_oft_rl supports multi-worker sharding.
            {"model_type": "oft",            "timeframe": "1m", "config": {"epochs": 5, "skip_sac": True}},
        ]
        task_ids = []
        for sym in symbols:
            safe = sym.replace("/", "_")
            for mc in model_configs:
                data_path = str(PROJECT_ROOT / "data" / "raw" / f"{safe}_{mc['timeframe']}.csv.gz")
                tid = self.submit_task({
                    **mc,
                    "symbol":      sym,
                    "data_path":   data_path,
                    "output_path": str(PROJECT_ROOT / "models"),
                })
                task_ids.append(tid)
        return task_ids


# ─── Singleton for dashboard embedding ───────────────────────────────────────

_orch_instance: Orchestrator | None = None
_orch_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    global _orch_instance
    if _orch_instance is None:
        with _orch_lock:
            if _orch_instance is None:
                _orch_instance = Orchestrator()
                _orch_instance.start()
    return _orch_instance


# ─── Standalone HTTP server ───────────────────────────────────────────────────

def _build_standalone_app(orch: Orchestrator):
    from flask import Flask, jsonify, request as freq
    app = Flask("orchestrator")

    @app.route("/api/cluster/status")
    def status():
        return jsonify(orch.get_status())

    # ── Phase 0 institutional upgrade: parquet store + ZMQ databus ────────
    @app.route("/api/parquet/status")
    def parquet_status():
        try:
            from src.database.parquet_store import get_store
            return jsonify(get_store().status())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/databus/stats")
    def databus_stats():
        try:
            from src.transport.data_bus import get_data_bus
            return jsonify(get_data_bus().stats())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/cluster/workers")
    def workers():
        return jsonify(orch.list_workers())

    @app.route("/api/cluster/tasks")
    def tasks():
        return jsonify(orch.list_tasks())

    @app.route("/api/cluster/submit", methods=["POST"])
    def submit():
        spec = freq.get_json(force=True) or {}
        tid  = orch.submit_task(spec)
        return jsonify({"ok": True, "task_id": tid})

    @app.route("/api/cluster/submit_all", methods=["POST"])
    def submit_all():
        body    = freq.get_json(force=True) or {}
        symbols = body.get("symbols")
        ids     = orch.submit_full_training_run(symbols)
        return jsonify({"ok": True, "task_ids": ids, "count": len(ids)})

    @app.route("/api/cluster/register", methods=["POST"])
    def register():
        orch.register_worker(freq.get_json(force=True) or {})
        return jsonify({"ok": True})

    @app.route("/api/cluster/task_update", methods=["POST"])
    def task_update():
        body = freq.get_json(force=True) or {}
        orch.update_task(
            body.get("task_id", ""),
            body.get("status", ""),
            node_id=body.get("node_id", ""),
            result=body.get("result"),
            error=body.get("error", ""),
        )
        return jsonify({"ok": True})

    @app.route("/api/cluster/task/<task_id>", methods=["DELETE"])
    def cancel(task_id):
        ok = orch.cancel_task(task_id)
        return jsonify({"ok": ok})

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="AI Trading — Training Orchestrator")
    parser.add_argument("--port", type=int, default=ORCH_PORT, help=f"HTTP port (default {ORCH_PORT})")
    args = parser.parse_args()

    orch = Orchestrator()
    orch.start()
    app = _build_standalone_app(orch)

    local_ip = _local_ip()
    logger.info("=" * 60)
    logger.info("Training Orchestrator — http://%s:%d", local_ip, args.port)
    logger.info("Workers connect with:")
    logger.info("  python -m src.training.distributed.worker --master http://%s:%d", local_ip, args.port)
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


def _local_ip() -> str:
    """Return the 192.168.0.x LAN IP if available, otherwise any non-loopback IP."""
    import socket as _sock
    try:
        import psutil
        for iface_addrs in psutil.net_if_addrs().values():
            for addr in iface_addrs:
                if addr.family == _sock.AF_INET and addr.address.startswith("192.168.0."):
                    return addr.address
    except Exception:
        pass
    try:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
