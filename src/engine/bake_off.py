"""
Sprint 0 §S0-2 — Model architecture bake-off harness.

Drives a structured comparison across (model, tf) cells using the KPI gate's
persisted Parquet runs. Output is a ranked cut list: rows with their KPIs,
relative ranks, and a recommendation ('keep' / 'retire' / 'review').

Usage:
    from src.engine.bake_off import run_bake_off
    ranked = run_bake_off(metric='wf_sharpe', top_n=10)
    # Returns a list of dicts sorted by `metric` descending.

The harness does NOT train anything itself — it consumes whatever the KPI
gate has logged so far. To run a fresh bake-off across all cells, the
operator triggers retrains via the cluster (or CIO Agent) first, then this
module reads the resulting Parquet.

CLI:
    python -m src.engine.bake_off --metric wf_sharpe --top 10
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS_DIR = PROJECT_ROOT / 'data' / 'training_runs'
CUT_LIST_PATH = PROJECT_ROOT / 'data' / 'bake_off_cut_list.json'

# Metrics where SMALLER is better (max-drawdown). All others: bigger=better.
INVERTED_METRICS = {'wf_max_dd'}


@dataclass
class BakeOffRow:
    model: str
    tf: str
    n_runs: int
    metric_value: float | None
    rank: int = 0
    recommendation: str = 'keep'  # keep | retire | review
    reasons: list[str] = field(default_factory=list)
    retired_by_kpi_gate: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def _read_latest_run(model: str, tf: str) -> dict | None:
    """Return the most-recent successful TrainingResult row for (model, tf)."""
    try:
        from src.engine import kpi_gate as kg
    except Exception:
        return None
    runs = kg.last_n_successful(model, tf, n=1)
    if not runs:
        return None
    r = runs[0]
    return {
        'wf_sharpe':       r.wf_sharpe,
        'wf_calmar':       r.wf_calmar,
        'wf_max_dd':       r.wf_max_dd,
        'wf_win_rate':     r.wf_win_rate,
        'wf_expectancy':   r.wf_expectancy,
        'wf_total_trades': r.wf_total_trades,
        'wf_acc':          r.wf_acc,
        'auc_roc':         r.auc_roc,
        'n_samples':       r.n_samples,
        'n_features':      r.n_features,
        'finished_at':     r.finished_at,
    }


def _read_n_runs(model: str, tf: str) -> int:
    """Count of successful runs persisted for (model, tf)."""
    try:
        from src.engine import kpi_gate as kg
    except Exception:
        return 0
    return len(kg.last_n_successful(model, tf, n=100))


def _is_retired(model: str, tf: str) -> bool:
    try:
        from src.engine import kpi_gate as kg
        return kg.is_retired(model, tf)
    except Exception:
        return False


def _enumerate_cells() -> list[tuple[str, str]]:
    """Read training_rules.json and yield every (model, tf) cell —
    applicable + experimental — that's potentially in the bake-off."""
    rules_path = PROJECT_ROOT / 'data' / 'training_rules.json'
    if not rules_path.exists():
        return []
    try:
        rules = json.loads(rules_path.read_text(encoding='utf-8')) or {}
    except Exception:
        return []
    cells: list[tuple[str, str]] = []
    for model, cfg in (rules.get('models') or {}).items():
        for tf in list(cfg.get('applicable_tfs') or []) + list(cfg.get('experimental_tfs') or []):
            cells.append((model, tf))
    return cells


def run_bake_off(
    metric: str = 'wf_sharpe',
    top_n: int = 20,
    retire_below_pct: float = 0.20,
) -> dict:
    """
    Compute the cut list. Returns:
      {
        'metric': <metric>,
        'rows':   [BakeOffRow as dict, ...],  // sorted best-first
        'cut_list': {
          'keep':    [...],
          'review':  [...],
          'retire':  [...],
        },
      }

    Recommendation rules:
      - retired_by_kpi_gate=True  → 'retire' (already auto-retired)
      - metric_value is None      → 'review' (no successful runs yet)
      - rank in bottom `retire_below_pct` of populated cells → 'retire'
      - otherwise → 'keep'
    """
    cells = _enumerate_cells()
    rows: list[BakeOffRow] = []

    for model, tf in cells:
        latest = _read_latest_run(model, tf)
        row = BakeOffRow(
            model=model, tf=tf,
            n_runs=_read_n_runs(model, tf),
            metric_value=(latest.get(metric) if latest else None),
            retired_by_kpi_gate=_is_retired(model, tf),
            extras=latest or {},
        )
        rows.append(row)

    # Sort: cells with values first (best metric to worst), then None rows.
    invert = metric in INVERTED_METRICS

    def _sort_key(r: BakeOffRow):
        if r.metric_value is None:
            return (1, 0.0)  # push to end
        # invert=True ⇒ smaller is better ⇒ negate so descending sort picks smallest
        v = -r.metric_value if invert else r.metric_value
        return (0, -v)  # within "has value", best first
    rows.sort(key=_sort_key)

    populated = [r for r in rows if r.metric_value is not None]
    n_pop = len(populated)
    cutoff_idx = int(n_pop * (1.0 - retire_below_pct))  # everything past here is bottom-X%

    for i, row in enumerate(rows):
        row.rank = i + 1
        if row.retired_by_kpi_gate:
            row.recommendation = 'retire'
            row.reasons.append('already retired by KPI gate')
        elif row.metric_value is None:
            row.recommendation = 'review'
            row.reasons.append(f'no successful runs (n_runs={row.n_runs})')
        elif i >= cutoff_idx and i < n_pop:
            row.recommendation = 'retire'
            row.reasons.append(
                f'bottom {retire_below_pct*100:.0f}% on {metric} '
                f'(rank {row.rank}/{n_pop})'
            )

    cut = {'keep': [], 'review': [], 'retire': []}
    result_rows = []
    for r in rows:
        d = {
            'model': r.model, 'tf': r.tf, 'rank': r.rank,
            'n_runs': r.n_runs,
            'metric': metric,
            'metric_value': r.metric_value,
            'recommendation': r.recommendation,
            'reasons': r.reasons,
            'retired_by_kpi_gate': r.retired_by_kpi_gate,
        }
        result_rows.append(d)
        cut[r.recommendation].append(f'{r.model}__{r.tf}')

    out = {
        'metric': metric,
        'retire_below_pct': retire_below_pct,
        'top_n': top_n,
        'rows':   result_rows[:top_n] if top_n else result_rows,
        'cut_list': cut,
    }

    # Persist for the dashboard / next session
    try:
        CUT_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        out_with_ts = dict(out)
        out_with_ts['generated_at'] = datetime.now(timezone.utc).isoformat()
        CUT_LIST_PATH.write_text(json.dumps(out_with_ts, indent=2), encoding='utf-8')
    except Exception as e:
        logger.warning('[bake_off] could not persist cut list: %s', e)

    return out


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser(description='Sprint 0 §S0-2 model bake-off')
    ap.add_argument('--metric', default='wf_sharpe',
                    choices=['wf_sharpe', 'wf_calmar', 'wf_acc', 'wf_win_rate',
                             'wf_expectancy', 'wf_total_trades', 'wf_max_dd', 'auc_roc'])
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--retire-pct', type=float, default=0.20)
    args = ap.parse_args()
    result = run_bake_off(metric=args.metric, top_n=args.top, retire_below_pct=args.retire_pct)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
