"""
FastAPI control plane (port 8100) — Phase 0.

Scope:
  - Inspect Parquet store (status, sizes, freshness per symbol)
  - Trigger one-off operations (ingest CSV, drop symbol)
  - Inspect ZeroMQ DataBus state
  - Health/ready probes for the cluster

This service complements (does not replace) the existing Flask orchestrator
on port 7700. The orchestrator continues to handle distributed training
job dispatch via REST. The FastAPI control plane handles
institutional-upgrade-specific operations that don't belong in the
orchestrator.

Run:
    uvicorn src.transport.control_api:app --host 0.0.0.0 --port 8100
or:
    python -m src.transport.control_api
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.parquet_store import ParquetStore, get_store
from src.transport.zmq_config import CONTROL_API_HOST, CONTROL_API_PORT
from src.transport.data_bus import get_data_bus

logger = logging.getLogger("control_api")


def _build_app():
    """Lazy import of FastAPI so unit tests can run without the dep."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class IngestRequest(BaseModel):
        csv_path: str
        symbol:   str
        skip_existing: bool = True

    class DropRequest(BaseModel):
        symbol: str

    app = FastAPI(title="AI Trading — Control Plane", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok":   True,
            "name": "control_api",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "ports": {"control_api": CONTROL_API_PORT},
        }

    @app.get("/parquet/status")
    def parquet_status() -> dict:
        return get_store().status()

    @app.get("/parquet/symbols")
    def parquet_symbols() -> dict:
        return {"symbols": get_store().list_symbols()}

    @app.get("/parquet/symbol/{symbol_safe}")
    def parquet_symbol(symbol_safe: str) -> dict:
        # symbol_safe uses underscore: BTC_USDT
        symbol = symbol_safe.replace("_", "/")
        return get_store().symbol_status(symbol).__dict__

    @app.post("/parquet/ingest")
    def parquet_ingest(req: IngestRequest) -> dict:
        path = Path(req.csv_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"csv_path not found: {path}")
        try:
            return get_store().ingest_csv(path, req.symbol, skip_existing=req.skip_existing)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/parquet/drop")
    def parquet_drop(req: DropRequest) -> dict:
        ok = get_store().drop_symbol(req.symbol)
        return {"ok": ok, "symbol": req.symbol}

    @app.get("/databus/stats")
    def databus_stats() -> dict:
        return get_data_bus().stats()

    @app.get("/")
    def root() -> dict:
        return {
            "service": "AI Trading — Control Plane",
            "endpoints": [
                "/health",
                "/parquet/status",
                "/parquet/symbols",
                "/parquet/symbol/{symbol_safe}",
                "/parquet/ingest  (POST)",
                "/parquet/drop    (POST)",
                "/databus/stats",
            ],
        }

    return app


# Module-level app for `uvicorn src.transport.control_api:app`
try:
    app = _build_app()
except Exception as exc:
    # If FastAPI is not installed, app stays None — tests can still import the module.
    logger.warning("FastAPI not available -- control_api app not built: %s", exc)
    app = None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if app is None:
        logger.error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn[standard]")
        return 1
    import uvicorn
    logger.info("Starting Control Plane on http://%s:%d", CONTROL_API_HOST, CONTROL_API_PORT)
    uvicorn.run(app, host=CONTROL_API_HOST, port=CONTROL_API_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
