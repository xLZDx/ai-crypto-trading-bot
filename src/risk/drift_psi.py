"""Phase 6 (2026-05-14) — F2 Concept Drift detection (PSI + Wasserstein).

Extends the existing z-test in src/risk/validators.py with two industry-
standard drift metrics:

  Population Stability Index (PSI)
    PSI = Σ (p_actual_i - p_expected_i) * ln(p_actual_i / p_expected_i)
    over the quantile bins of the baseline feature distribution.
    Banked threshold lookup table:
      PSI < 0.10  → no drift
      0.10–0.25   → moderate drift (WARN)
      ≥ 0.25      → significant drift (PAUSE)
    These are the canonical credit-risk thresholds; tighter than the
    0.25/0.50 ml-engineer flagged as "too lax for HFT signals".

  Wasserstein distance
    Earth-mover's distance between baseline and live empirical CDFs.
    Used as a secondary metric for shape-sensitive drift (a bimodal
    actual distribution can match the baseline mean+variance but
    Wasserstein-diverge). Computed via scipy.stats.wasserstein_distance
    on the inverse-CDF samples derived from baseline bin edges.

Why no `evidently` dependency
-----------------------------
ml-engineer's first preference was `evidently`. After review, PSI and
Wasserstein over our existing baseline schema can be computed in ~30
lines of numpy/scipy without adding a 30 MB dependency. The dashboard
rendering layer (operator-facing drift report) IS a candidate for
evidently if/when we need the HTML report — flagged as Phase 6b.

Enforcement modes (matches MODEL_HMAC_ENFORCEMENT pattern):
  LLM_DRIFT_PAUSE=enforce  → PSI ≥ 0.25 raises DriftPauseError
  LLM_DRIFT_PAUSE=warn (default) → all drift logged WARNING, no pause
  LLM_DRIFT_PAUSE=off      → drift checks skipped entirely

Per-feature scope
-----------------
The hard-pause trigger only fires on features in DRIFT_HARD_FEATURES
(engineered, label-correlated). Raw OHLCV drift is regime change, not
model breakage — warn only. Operator overrides via the env var
DRIFT_HARD_FEATURES_EXTRA (comma-separated additions).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# Engineered, label-correlated features whose drift signals genuine
# model degradation rather than mere regime change.
DRIFT_HARD_FEATURES: frozenset[str] = frozenset({
    "ofi_z", "ofi", "funding_z",
    "macd_hist", "macd_signal",
    "frac_diff_d40",
    "atr_14",
    "rsi_14",
    "kc_pos", "kc_width",
    "don_pos_20",
    "vwap_dist",
    "trend_strength", "vol_regime",
    "taker_buy_ratio",
})


PSI_WARN_THRESHOLD = 0.10
PSI_PAUSE_THRESHOLD = 0.25
WASSERSTEIN_WARN_THRESHOLD = 0.15  # relative — see compute_wasserstein()


_MODE_ENV = "LLM_DRIFT_PAUSE"
_MODE_ENFORCE = "enforce"
_MODE_WARN = "warn"
_MODE_OFF = "off"
_VALID_MODES = frozenset({_MODE_ENFORCE, _MODE_WARN, _MODE_OFF})


class DriftPauseError(RuntimeError):
    """Raised when LLM_DRIFT_PAUSE=enforce and a hard-feature exceeds the
    pause threshold. The bot's main loop / cluster orchestrator catches
    this and halts new trade signals until the operator clears the flag."""


@dataclass
class DriftFinding:
    feature: str
    psi: float
    wasserstein_rel: float
    severity: str  # "ok" | "warn" | "pause"
    is_hard_feature: bool = False
    note: str = ""


@dataclass
class DriftReport:
    findings: list[DriftFinding] = field(default_factory=list)
    pause_triggered: bool = False
    mode: str = "warn"

    def by_severity(self, severity: str) -> list[DriftFinding]:
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pause_triggered": self.pause_triggered,
            "ok_count": len(self.by_severity("ok")),
            "warn_count": len(self.by_severity("warn")),
            "pause_count": len(self.by_severity("pause")),
            "findings": [
                {"feature": f.feature, "psi": round(f.psi, 4),
                 "wasserstein_rel": round(f.wasserstein_rel, 4),
                 "severity": f.severity, "is_hard": f.is_hard_feature,
                 "note": f.note}
                for f in self.findings
            ],
        }


def _load_mode() -> str:
    raw = (os.environ.get(_MODE_ENV) or _MODE_WARN).strip().lower()
    if raw in _VALID_MODES:
        return raw
    logger.warning("%s=%r invalid; defaulting to %r", _MODE_ENV, raw, _MODE_WARN)
    return _MODE_WARN


def _hard_features() -> frozenset[str]:
    """Return the effective hard-feature set: built-in + env extras."""
    extras = (os.environ.get("DRIFT_HARD_FEATURES_EXTRA") or "").strip()
    if extras:
        extra_set = {x.strip() for x in extras.split(",") if x.strip()}
        return DRIFT_HARD_FEATURES | extra_set
    return DRIFT_HARD_FEATURES


def compute_psi(
    baseline_props: list[float],
    baseline_edges: list[float],
    actual_values: np.ndarray | list[float],
    *,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index.

    PSI = Σ (p_actual - p_expected) * ln(p_actual / p_expected) over the
    bins defined by `baseline_edges`. Bin counts in `actual_values` are
    floor-clipped to `epsilon` so empty bins don't blow up the log.

    Returns 0.0 if the baseline has fewer than 2 bins (constant feature).
    """
    if baseline_edges is None or len(baseline_edges) < 2:
        return 0.0
    actual = np.asarray(actual_values, dtype=float)
    if len(actual) == 0:
        return 0.0
    edges = np.asarray(baseline_edges, dtype=float)
    # Clip the actuals into the baseline's bin range so np.histogram
    # accounts for out-of-range values (otherwise a fully-shifted
    # distribution silently returns total=0 and PSI=0, masking the
    # very drift we're trying to detect). Values below the first edge
    # land in bin 0; values above the last edge land in the last bin.
    eps = max(1e-9, float(edges[-1] - edges[0]) * 1e-6)
    clipped = np.clip(actual, edges[0] + eps, edges[-1] - eps)
    counts, _ = np.histogram(clipped, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    actual_props = counts / total
    expected = np.clip(np.asarray(baseline_props, dtype=float), epsilon, None)
    actual_props = np.clip(actual_props, epsilon, None)
    # Align lengths (baseline bins == actual bins by construction of np.histogram).
    if len(expected) != len(actual_props):
        m = min(len(expected), len(actual_props))
        expected = expected[:m]
        actual_props = actual_props[:m]
    psi = float(np.sum((actual_props - expected) * np.log(actual_props / expected)))
    return psi


def compute_wasserstein_relative(
    baseline_edges: list[float],
    baseline_props: list[float],
    actual_values: np.ndarray | list[float],
) -> float:
    """Wasserstein distance normalized by the baseline's IQR so the
    threshold is dimensionless. Uses bin midpoints as the baseline
    sample proxy.

    Why "relative": the raw Wasserstein scales with the feature's unit
    (price-in-USD vs. z-score). Dividing by IQR (q75 - q25 of the
    baseline) lets us use a single threshold across all features.
    """
    if baseline_edges is None or len(baseline_edges) < 2:
        return 0.0
    actual = np.asarray(actual_values, dtype=float)
    if len(actual) == 0:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:  # pragma: no cover — scipy is in the trading-bot venv
        return 0.0
    edges = np.asarray(baseline_edges, dtype=float)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    props = np.asarray(baseline_props, dtype=float)
    if len(midpoints) != len(props):
        m = min(len(midpoints), len(props))
        midpoints = midpoints[:m]
        props = props[:m]
    raw_wd = wasserstein_distance(midpoints, actual, u_weights=props)
    # Normalize by baseline IQR.
    iqr = float(edges[-1] - edges[0])  # approximate; quantile-edge span
    if iqr <= 0:
        return float(raw_wd)
    return float(raw_wd / iqr)


def check_drift(
    baseline: dict[str, dict],
    actual_df,
    *,
    hard_features: Iterable[str] | None = None,
) -> DriftReport:
    """Compare a live feature DataFrame against the persisted baseline.

    `baseline` is the dict returned by drift_baseline.load_baseline (or a
    test fixture). Each entry must carry `bin_edges` + `bin_props` for
    PSI; entries missing these are skipped (rolled-up as soft-warn).

    Returns a DriftReport whose `pause_triggered` flag is True iff at
    least one HARD-feature exceeded PSI_PAUSE_THRESHOLD AND
    LLM_DRIFT_PAUSE=enforce.
    """
    mode = _load_mode()
    rep = DriftReport(mode=mode)
    if mode == _MODE_OFF:
        return rep
    if actual_df is None or len(actual_df) == 0 or not baseline:
        return rep

    hard_set = set(hard_features) if hard_features is not None else _hard_features()

    for feat, stats in baseline.items():
        if feat not in actual_df.columns:
            continue
        series = actual_df[feat].dropna()
        if len(series) < 30:  # too small to detect drift reliably
            continue
        edges = stats.get("bin_edges")
        props = stats.get("bin_props")
        if not edges or not props:
            continue
        psi = compute_psi(props, edges, series.values)
        wd = compute_wasserstein_relative(edges, props, series.values)
        is_hard = feat in hard_set
        if psi >= PSI_PAUSE_THRESHOLD:
            severity = "pause" if is_hard else "warn"
        elif psi >= PSI_WARN_THRESHOLD or wd >= WASSERSTEIN_WARN_THRESHOLD:
            severity = "warn"
        else:
            severity = "ok"
        rep.findings.append(DriftFinding(
            feature=feat, psi=psi, wasserstein_rel=wd,
            severity=severity, is_hard_feature=is_hard,
        ))

    # Aggregate decision.
    pause_findings = rep.by_severity("pause")
    if pause_findings and mode == _MODE_ENFORCE:
        rep.pause_triggered = True
        feat_names = ", ".join(f.feature for f in pause_findings[:5])
        msg = (f"DriftPauseError: {len(pause_findings)} hard-feature(s) "
               f"PSI ≥ {PSI_PAUSE_THRESHOLD} → {feat_names} "
               f"(LLM_DRIFT_PAUSE=enforce)")
        logger.critical(msg)
        raise DriftPauseError(msg)

    for f in rep.findings:
        if f.severity in ("warn", "pause"):
            logger.warning(
                "[drift] %s PSI=%.3f WD_rel=%.3f severity=%s hard=%s",
                f.feature, f.psi, f.wasserstein_rel, f.severity, f.is_hard_feature,
            )

    return rep


__all__ = [
    "DRIFT_HARD_FEATURES", "DriftFinding", "DriftReport", "DriftPauseError",
    "PSI_WARN_THRESHOLD", "PSI_PAUSE_THRESHOLD",
    "compute_psi", "compute_wasserstein_relative", "check_drift",
]
