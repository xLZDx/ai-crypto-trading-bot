"""
src.data_governance — Phase 8.

A unified framework for managing every external data source the bot uses
to train models. The goals are:

  • One config (`data/data_governance.json`) controls which sources are
    enabled, their priority, and their poll interval.
  • Each source implements `DataSourceConnector` — a thin contract with
    `pull_history(...)`, `realtime_loop(...)`, and `is_available()`.
  • All HTTP requests go through `rate_limiter.get_limiter(host)` so
    concurrent connectors share a global budget.
  • All data is stored at the DB level (QuestDB hot path + Parquet cold
    path) — never in long-lived process memory.
  • Connectors that need an API key are *graceful*: missing key → log
    once, skip — never crash the orchestrator.

Public API:
    from src.data_governance import DataSourceConnector, REGISTRY, GovernanceConfig
"""
from .base import DataSourceConnector
from .config import GovernanceConfig, DEFAULT_CONFIG_PATH
from .registry import REGISTRY, register, list_sources

# Side-effect import: each module under connectors/ runs its @register
# decorator at import time. Without this line the REGISTRY stays empty
# and the dashboard's Strategies tab renders "No data sources registered".
from . import connectors as _connectors  # noqa: F401

__all__ = [
    "DataSourceConnector", "GovernanceConfig", "DEFAULT_CONFIG_PATH",
    "REGISTRY", "register", "list_sources",
]
