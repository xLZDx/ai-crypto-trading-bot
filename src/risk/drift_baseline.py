"""
Distribution drift baseline collection (Sprint 0 §S0-1 follow-on).

Produces and persists per-feature {mean, std, q05, q95, sample_size} from a
known-good feature matrix. The ValidationGate's drift check uses these as the
reference distribution for its z-test on feature mean shift.

Usage:
    from src.risk.drift_baseline import save_baseline, load_baseline
    save_baseline(model_key='meta', timeframe='1h', feature_df=df)
    ref = load_baseline('meta', '1h')   # → {col_name: {mean, std, ...}}

Persistence: one JSON file per (model, tf) at
    data/risk/drift_baselines/<model>__<tf>.json

The baseline should be refreshed when:
  - A model is retrained and APPROVE'd by ML Engineer + KPI gate
  - The operator manually triggers a baseline reset (future endpoint)

Stale baselines are detected by `baseline_age_days` — the loader returns
None for baselines older than the configured max_age_days, forcing the
drift check to skip rather than fire false positives on regime changes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = PROJECT_ROOT / 'data' / 'risk' / 'drift_baselines'

# Per-baseline freshness — older than this and we don't enforce drift.
DEFAULT_MAX_AGE_DAYS: int = 30


def _baseline_path(model_key: str, timeframe: str) -> Path:
    safe = f'{model_key}__{timeframe}'
    return BASELINES_DIR / f'{safe}.json'


def save_baseline(model_key: str, timeframe: str, feature_df: pd.DataFrame,
                  n_bins: int = 10) -> dict:
    """
    Compute per-column distribution stats from `feature_df` and persist.

    Per-feature payload:
      - mean, std, q05, q95, n (the original z-test inputs)
      - bin_edges, bin_props (Phase 6, 2026-05-14) — quantile bin edges
        + proportion per bin so the PSI drift check can compare a live
        sample against this empirical distribution.

    Returns the saved dict (so the caller can include it in the meta JSON
    for audit). Empty dict on failure.
    """
    if feature_df is None or feature_df.empty:
        logger.warning("[drift_baseline] cannot save: empty DataFrame")
        return {}

    import numpy as np  # imported locally so the module stays import-cheap
    summary: dict[str, dict] = {}
    for col in feature_df.columns:
        series = pd.to_numeric(feature_df[col], errors='coerce').dropna()
        if len(series) < 10:
            continue
        entry = {
            'mean': float(series.mean()),
            'std':  float(series.std()),
            'q05':  float(series.quantile(0.05)),
            'q95':  float(series.quantile(0.95)),
            'n':    int(len(series)),
        }
        # Phase 6 — quantile bins for PSI. We add a small ε to the last bin
        # edge so np.digitize handles the maximum value cleanly. Constant
        # features (all-same value) get a single-bin histogram and are
        # effectively skipped by the PSI check downstream.
        try:
            if series.nunique() <= 1:
                entry['bin_edges'] = [float(series.min()), float(series.min()) + 1e-9]
                entry['bin_props'] = [1.0]
            else:
                quantiles = np.linspace(0, 1, n_bins + 1)
                edges = series.quantile(quantiles).unique()
                edges = np.sort(edges)
                if len(edges) < 2:
                    edges = np.array([float(series.min()), float(series.max()) + 1e-9])
                # Last edge bumped so digitize includes the max value.
                edges[-1] = edges[-1] + 1e-9
                counts, _ = np.histogram(series.values, bins=edges)
                total = counts.sum()
                props = (counts / total).tolist() if total else [0.0] * (len(edges) - 1)
                entry['bin_edges'] = [float(e) for e in edges]
                entry['bin_props'] = props
        except Exception as e:
            logger.debug("[drift_baseline] bin build failed for %s: %s", col, e)
        summary[col] = entry

    if not summary:
        logger.warning("[drift_baseline] cannot save: no usable numeric columns")
        return {}

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_key': model_key,
        'timeframe': timeframe,
        'saved_at':  datetime.now(timezone.utc).isoformat(),
        'features':  summary,
        'feature_count': len(summary),
    }
    path = _baseline_path(model_key, timeframe)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    logger.info("[drift_baseline] saved %d feature stats -> %s", len(summary), path.name)
    return payload


def load_baseline(
    model_key: str,
    timeframe: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict[str, dict] | None:
    """
    Return the per-feature stats dict for the drift check, or None if:
      - no baseline exists for (model, tf)
      - baseline is older than max_age_days
      - file is malformed

    The ValidationGate.run() passes this to its _check_drift().
    """
    path = _baseline_path(model_key, timeframe)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning("[drift_baseline] could not parse %s: %s", path.name, e)
        return None
    saved_at = payload.get('saved_at')
    if saved_at:
        try:
            ts = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days > max_age_days:
                logger.info(
                    "[drift_baseline] %s is %.1fd old (> %d) -- skipping drift check",
                    path.name, age_days, max_age_days,
                )
                return None
        except Exception:
            pass  # if we can't parse the timestamp, fall through and use it
    features = payload.get('features')
    if not isinstance(features, dict):
        return None
    return features


def baseline_age_days(model_key: str, timeframe: str) -> float | None:
    """Helper for the dashboard tile — how stale is the baseline?"""
    path = _baseline_path(model_key, timeframe)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        ts = datetime.fromisoformat(payload['saved_at'].replace('Z', '+00:00'))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    except Exception:
        return None
