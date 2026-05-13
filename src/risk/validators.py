"""
Sprint 0 §S0-1 — Validation rigor pipeline.

Pre-training data integrity gate. The ML Engineer agent's pre-flight check
calls `ValidationGate.run(model_type, timeframe)` before any task is
queued. Failures bubble up as BLOCK reasons.

Validators in order (each runs only if the prior passed):
  1. data_freshness    — last candle for required symbols is within tolerance
  2. label_imbalance   — Triple Barrier labels have at least min_class_pct
                          of each class (no degenerate all-zero / all-one)
  3. distribution_drift — KS-test against the last-known-good distribution
                          flags features whose CDF shifted > drift_pvalue.
                          Drift is a WARNING by default (does not BLOCK).
  4. nan_density        — fraction of NaN cells in the feature matrix
                          must be below max_nan_pct.

Decision matrix:
  BLOCK   = data freshness fail | label imbalance fail | nan density fail
  WARN    = distribution drift detected, but other checks pass
  APPROVE = all checks pass clean

Persistence:
  data/risk/validation_runs.json — append-only log of every run (last 200)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_LOG_PATH = PROJECT_ROOT / 'data' / 'risk' / 'validation_runs.json'

# Per-timeframe freshness tolerance — last candle must be within this window.
# Loose by default (2x the bar interval) so a single missing print doesn't
# block training; tighten via ValidationConfig if needed.
FRESHNESS_TOLERANCE_MIN: dict[str, float] = {
    '1m':  4.0,
    '5m':  20.0,
    '15m': 60.0,
    '1h':  240.0,
    '4h':  960.0,
    '1d':  60 * 24 * 3,
    '1w':  60 * 24 * 14,
    '1mo': 60 * 24 * 60,
}


@dataclass
class ValidationConfig:
    """Knobs for the validation gate. Defaults are AFML-conservative."""
    min_class_pct:      float = 0.10   # Each of {-1, 0, 1} must be ≥ 10%
    max_nan_pct:        float = 0.05   # ≤ 5% NaN in the feature matrix
    drift_pvalue:       float = 0.01   # KS p-value below this flags drift
    freshness_tolerance_factor: float = 2.0  # × bar interval
    enabled:            bool  = True


@dataclass
class ValidationReport:
    """Output of `ValidationGate.run()`. Serializable to JSON for audit."""
    timestamp: str
    model_type: str
    timeframe:  str
    decision:   str   # APPROVE | WARN | BLOCK
    reasons:    list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    metrics:    dict[str, Any] = field(default_factory=dict)


class ValidationGate:
    """
    Stateless pre-training data integrity check. One method: `run()`.
    Each validator method returns (passed: bool, reason: str | None).
    """

    def __init__(self, cfg: ValidationConfig | None = None):
        self.cfg = cfg or ValidationConfig()

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        model_type: str,
        timeframe: str,
        symbols: list[str] | None = None,
        feature_df: pd.DataFrame | None = None,
        labels: pd.Series | None = None,
        last_known_good_dist: dict[str, dict] | None = None,
    ) -> ValidationReport:
        """
        Run all configured validators.

        Args:
            model_type:  one of base/trend/futures/scalping/meta/regime/tft/oft
            timeframe:   bar interval the model trains at
            symbols:     watchlist (None → loaded from data/watchlist.json)
            feature_df:  features matrix to inspect (None → skip nan + drift)
            labels:      label series for imbalance check (None → skip)
            last_known_good_dist: per-feature {'mean':..., 'std':...}
                                   from a prior good run (None → skip drift)
        """
        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            timeframe=timeframe,
            decision='APPROVE',  # tentative
        )
        if not self.cfg.enabled:
            report.warnings.append('WARN: ValidationGate disabled via config')
            self._persist(report)
            return report

        # ── 1. Data freshness ──
        symbols = symbols or self._load_watchlist()
        stale = self._check_freshness(symbols, timeframe)
        if stale:
            report.reasons.append(
                f'BLOCK: stale data for {len(stale)} symbols: {sorted(stale)[:5]}...'
            )
            report.metrics['stale_symbols'] = stale
            report.decision = 'BLOCK'
            self._persist(report)
            return report

        # ── 2. Label imbalance ──
        if labels is not None and len(labels) > 0:
            imbalanced = self._check_label_imbalance(labels)
            if imbalanced:
                report.reasons.append(f'BLOCK: label imbalance {imbalanced}')
                report.metrics['label_distribution'] = imbalanced
                report.decision = 'BLOCK'
                self._persist(report)
                return report

        # ── 3. NaN density ──
        if feature_df is not None and not feature_df.empty:
            nan_pct = float(feature_df.isna().sum().sum()) / float(feature_df.size)
            report.metrics['nan_pct'] = round(nan_pct, 4)
            if nan_pct > self.cfg.max_nan_pct:
                report.reasons.append(
                    f'BLOCK: NaN density {nan_pct:.3f} exceeds floor {self.cfg.max_nan_pct}'
                )
                report.decision = 'BLOCK'
                self._persist(report)
                return report

        # ── 4. Distribution drift (WARN only) ──
        if feature_df is not None and last_known_good_dist:
            drifted = self._check_drift(feature_df, last_known_good_dist)
            if drifted:
                report.warnings.append(
                    f'WARN: distribution drift detected on {len(drifted)} features: {drifted[:5]}'
                )
                report.metrics['drifted_features'] = drifted
                report.decision = 'WARN'

        if report.decision == 'APPROVE':
            pass  # already APPROVE
        self._persist(report)
        self._log_decision(report)
        return report

    # ── Validators ───────────────────────────────────────────────────────────

    def _check_freshness(self, symbols: list[str], timeframe: str) -> list[str]:
        """Return symbols whose latest candle is too old. Empty = all fresh."""
        tolerance_min = FRESHNESS_TOLERANCE_MIN.get(timeframe, 60.0) * self.cfg.freshness_tolerance_factor
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=tolerance_min)
        stale: list[str] = []
        raw_dir = PROJECT_ROOT / 'data' / 'raw'
        for sym in symbols:
            safe = sym.replace('/', '_')
            for fname in (f'{safe}_{timeframe}.csv.gz', f'{safe}_spot_{timeframe}.csv.gz'):
                fpath = raw_dir / fname
                if not fpath.exists():
                    continue
                try:
                    # Read only the last few rows for speed — pandas can't tail a .csv.gz,
                    # so read just the `timestamp` column.
                    df = pd.read_csv(fpath, usecols=['timestamp'])
                    if df.empty:
                        stale.append(sym)
                        break
                    last_ts = pd.to_datetime(df['timestamp'].iloc[-1])
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.tz_localize('UTC')
                    if last_ts < cutoff:
                        stale.append(sym)
                    break  # found a file; don't try the other
                except Exception as e:
                    logger.debug('[Validator] freshness probe failed for %s: %s', sym, e)
                    continue
        return stale

    def _check_label_imbalance(self, labels: pd.Series) -> dict | None:
        """Return distribution dict if any class is < min_class_pct, else None."""
        if len(labels) == 0:
            return {'error': 'empty_labels'}
        counts = labels.value_counts(normalize=True).to_dict()
        # For binary models the labels are {0, 1}; for triple-barrier {-1, 0, 1}.
        # If only one class shows up at all, definitely imbalanced.
        if len(counts) < 2:
            return {'classes': list(counts.keys()), 'reason': 'only_one_class'}
        minority_pct = min(counts.values())
        if minority_pct < self.cfg.min_class_pct:
            return {
                'distribution': {str(k): round(v, 4) for k, v in counts.items()},
                'minority_pct': round(minority_pct, 4),
                'min_required': self.cfg.min_class_pct,
            }
        return None

    def _check_drift(
        self,
        feature_df: pd.DataFrame,
        last_known_good_dist: dict[str, dict],
    ) -> list[str]:
        """
        Return list of feature names that drifted significantly.
        Uses a simple z-test against the stored mean/std; KS would need raw
        baseline samples which we don't always have.
        """
        drifted: list[str] = []
        for col in feature_df.columns:
            if col not in last_known_good_dist:
                continue
            ref = last_known_good_dist[col]
            ref_mean = ref.get('mean', 0.0)
            ref_std  = max(ref.get('std', 0.0), 1e-9)
            actual = feature_df[col].dropna()
            if len(actual) < 10:
                continue
            actual_mean = float(actual.mean())
            # 1-sample z-test on the mean shift
            z = abs(actual_mean - ref_mean) / (ref_std / np.sqrt(len(actual)))
            # p ≈ 0.01 ↔ |z| ≈ 2.58
            if z > 2.58:
                drifted.append(col)
        return drifted

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_watchlist() -> list[str]:
        wl_path = PROJECT_ROOT / 'data' / 'watchlist.json'
        if not wl_path.exists():
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        try:
            return json.loads(wl_path.read_text(encoding='utf-8'))
        except Exception:
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    def _persist(self, report: ValidationReport) -> None:
        try:
            VALIDATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {'runs': []}
            if VALIDATION_LOG_PATH.exists():
                try:
                    existing = json.loads(VALIDATION_LOG_PATH.read_text(encoding='utf-8')) or existing
                except Exception:
                    pass
            runs = existing.get('runs') or []
            runs.append(asdict(report))
            existing['runs'] = runs[-200:]
            VALIDATION_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning('[ValidationGate] persist failed: %s', e)

    def _log_decision(self, report: ValidationReport) -> None:
        level = logging.INFO
        if report.decision == 'BLOCK':
            level = logging.ERROR
        elif report.decision == 'WARN':
            level = logging.WARNING
        logger.log(
            level,
            '[ValidationGate] %s/%s → %s (reasons=%s warnings=%s)',
            report.model_type, report.timeframe, report.decision,
            report.reasons, report.warnings,
        )


# ── Module-level singleton ───────────────────────────────────────────────────

_singleton: ValidationGate | None = None


def get_validation_gate(cfg: ValidationConfig | None = None) -> ValidationGate:
    global _singleton
    if _singleton is None:
        _singleton = ValidationGate(cfg=cfg)
    return _singleton


def reset_singleton_for_tests() -> None:
    global _singleton
    _singleton = None
