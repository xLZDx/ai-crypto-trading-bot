"""
Shared protocol types for the distributed training cluster.

All communication is JSON over HTTP.  No external dependencies beyond
what's already in requirements.txt (requests, flask).

Task lifecycle:
  PENDING → RUNNING → DONE | FAILED
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    BTC_RF       = "btc_rf"
    TREND        = "trend"
    SCALPING     = "scalping"
    META_LABELER = "meta_labeler"
    FUTURES_SHORT = "futures_short"
    TFT          = "tft"
    OFT          = "oft"
    REGIME       = "regime"
    GARCH        = "garch"
    CUSTOM       = "custom"


@dataclass
class TrainingTask:
    task_id:     str
    model_type:  str          # ModelType value
    symbol:      str          # e.g. BTC/USDT
    timeframe:   str          # e.g. 1m
    config:      dict         # hyperparams, seq_len, etc.
    data_path:   str          # UNC or local path to training data on master
    output_path: str          # where worker writes model + metrics
    status:      str = TaskStatus.PENDING.value
    assigned_to: str = ""     # worker node_id
    created_at:  str = field(default_factory=lambda: _now_iso())
    started_at:  str = ""
    finished_at: str = ""
    result:      dict = field(default_factory=dict)
    error:       str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkerInfo:
    node_id:      str
    hostname:     str
    ip:           str
    port:         int
    gpu_name:     str   # e.g. "RTX 2800" or "CPU only"
    gpu_vram_gb:  float
    cpu_cores:    int
    ram_gb:       float
    cuda_available: bool
    status:       str = "idle"       # idle | busy | error
    current_task: str = ""
    last_seen:    str = field(default_factory=lambda: _now_iso())
    tasks_done:   int = 0
    tasks_failed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
