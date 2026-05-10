"""
src.transport — institutional-upgrade transport layer (Phase 0).

Two planes:
  - Control plane : FastAPI on port 8100 (REST commands, low volume)
  - Data plane    : ZeroMQ (PUB/SUB orderflow, PUSH/PULL training batches)

Designed so a later swap to Kafka touches only `data_bus.py`. Callers see the
same `publish_orderflow / subscribe_orderflow / push_batch / pull_batch` API.
"""
from .zmq_config import (
    ORDERFLOW_PORT,
    TRAINING_BATCH_PORT,
    CONTROL_FANOUT_PORT,
    CONTROL_API_PORT,
    bind_addr,
    connect_addr,
)
from .data_bus import DataBus, get_data_bus

__all__ = [
    "ORDERFLOW_PORT",
    "TRAINING_BATCH_PORT",
    "CONTROL_FANOUT_PORT",
    "CONTROL_API_PORT",
    "bind_addr",
    "connect_addr",
    "DataBus",
    "get_data_bus",
]
