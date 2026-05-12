"""
ZeroMQ + FastAPI port allocation for the institutional upgrade.

Port map (see docs/INSTITUTIONAL_UPGRADE_PLAN.md):
  5000   Dashboard (Flask, existing)
  7700   Orchestrator REST (Flask, existing)
  7701   Worker REST       (Flask, existing)
  8100   Control plane     (FastAPI, NEW — Phase 0)
  5555   Orderflow PUB/SUB (ZeroMQ, NEW — Phase 0)
  5556   Training batch PUSH/PULL (ZeroMQ, NEW — Phase 0)
  5557   Control fanout PUB/SUB (ZeroMQ, NEW — Phase 0)
"""
from __future__ import annotations

import os

# ── ZeroMQ data plane ─────────────────────────────────────────────────────────
ORDERFLOW_PORT       = int(os.getenv("ZMQ_ORDERFLOW_PORT",       "5555"))
TRAINING_BATCH_PORT  = int(os.getenv("ZMQ_TRAINING_BATCH_PORT",  "5556"))
CONTROL_FANOUT_PORT  = int(os.getenv("ZMQ_CONTROL_FANOUT_PORT",  "5557"))

# ── FastAPI control plane ────────────────────────────────────────────────────
# 2026-05-12 Phase A2 — default bind to localhost. The control plane
# is consumed only by same-host clients (the dashboard JS and local
# scripts). Phase A11 schedules deletion of this service entirely;
# until then, the safe default prevents LAN exposure. To restore
# 0.0.0.0 binding temporarily, set CONTROL_API_HOST in .env.
CONTROL_API_HOST = os.getenv("CONTROL_API_HOST", "127.0.0.1")
CONTROL_API_PORT = int(os.getenv("CONTROL_API_PORT", "8100"))

# ── Master / cluster ─────────────────────────────────────────────────────────
# Workers connect here for ZeroMQ data-plane sockets.
DATA_BUS_HOST = os.getenv("DATA_BUS_HOST", "")  # empty = auto-detect on master


def bind_addr(port: int, host: str = "*") -> str:
    """Server-side bind address. tcp://*:5555"""
    return f"tcp://{host}:{port}"


def connect_addr(port: int, host: str | None = None) -> str:
    """Client-side connect address. tcp://192.168.0.10:5555"""
    h = host or DATA_BUS_HOST or "127.0.0.1"
    return f"tcp://{h}:{port}"
