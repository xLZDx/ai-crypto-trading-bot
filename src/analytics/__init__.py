"""
src.analytics — Phase 9.

Decision-support layer that sits on top of QuestDB + Parquet. Provides:
  • DataLens — unified time-aligned query API joining OHLCV + funding + news
    + macro indicators across all configured sources.
  • DecisionMetrics — pre-aggregated feature/regime/sentiment summaries
    used by the trainer, the backtester, and the live bot.

Both classes are read-only — they never write to the DB. Storage stays
DB-level (per the user's data-governance principle).
"""
from .data_lens import DataLens
from .decision_metrics import DecisionMetrics, DecisionSummary

__all__ = ["DataLens", "DecisionMetrics", "DecisionSummary"]
