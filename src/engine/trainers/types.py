"""src.engine.trainers.types — typed contract for trainer output.

Sprint 1a R1 Step 1: introduces `TrainingResult` so every trainer (regardless
of underlying implementation — RF, HistGBT, TFT/Darts, OFT/RL, GMM, meta)
produces a comparable output shape. Sprint 1a R2 builds the KPI gate on top
of this; R3's "Model Comparison" dashboard reads the same fields.

Fields are NULLABLE where applicable so trainers that don't compute a given
metric (e.g. regime classifier has no walk-forward folds) can return None
without breaking the contract. Consumers must treat None as "metric not
applicable to this trainer" — not as "metric failed to compute".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TrainingResult:
    """Sprint 1a R1 — standardized output for any trainer.

    See src.engine.trainers.train_<key>.train() — each wrapper returns one
    of these. The fields below are the SUPERSET; individual trainers
    populate what's applicable.
    """
    # Identity
    model_key:     str                  # 'base' | 'trend' | ... | 'regime'
    tf:            str                  # '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w'
    symbol:        str = "BTC/USDT"     # primary symbol (some trainers train multi-symbol)

    # Lifecycle
    started_at:    float = 0.0          # unix seconds
    finished_at:   float = 0.0          # unix seconds
    elapsed_s:     float = 0.0
    artifact_path: str | None = None    # path to persisted .joblib / .pt artifact

    # Data stats
    n_samples:     int | None = None
    n_features:    int | None = None
    n_iterations:  int | None = None    # boosting rounds (HistGBT) / epochs (neural)

    # KPI block — same fields across every trainer for apples-to-apples
    # comparison. Sprint 1a R2 KPI gate reads these.
    wf_sharpe:        float | None = None   # walk-forward Sharpe ratio
    wf_calmar:        float | None = None   # Sharpe / max DD
    wf_max_dd:        float | None = None   # walk-forward max drawdown (fraction)
    wf_win_rate:      float | None = None   # walk-forward fold win rate (%)
    wf_expectancy:    float | None = None   # mean PnL per trade, net of costs
    wf_total_trades:  int   | None = None   # count across folds
    wf_acc:           float | None = None   # walk-forward classification accuracy (%)
    auc_roc:          float | None = None   # ROC-AUC (probabilistic classifiers only)
    test_acc:         float | None = None   # held-out test accuracy
    long_acc:         float | None = None   # precision when predicting long (%)
    short_acc:        float | None = None   # precision when predicting short (%)
    win_precision:    float | None = None   # meta-labeler precision at gate threshold (%)

    # Outcome
    error:         str | None = None    # populated on failure; None when ok
    cancelled:     bool = False         # True if operator killed mid-run

    # Per-trainer free-form extras — never relied on by KPI gate, useful for
    # operator drill-down. Keep small (<1KB) to avoid cluttering Parquet rows.
    extras:        dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff trainer ran to completion without error or cancellation."""
        return self.error is None and not self.cancelled

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / Parquet row. Empty extras are dropped."""
        d = asdict(self)
        if not d.get('extras'):
            d.pop('extras', None)
        return d
