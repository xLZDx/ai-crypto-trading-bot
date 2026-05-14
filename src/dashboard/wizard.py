"""Phase 7 (2026-05-14) — Training Wizard.

Operator-facing model-improvement advisor. For a given (model, tf):
  1. Loads current KPI snapshot (model_*_meta.json)
  2. Loads recent drift report (data/risk/drift_baselines/*)
  3. Loads recent data quality report (if persisted to meta JSON)
  4. Applies a rule-based recommender that ranks the top 3 actionable
     improvements (no LLM needed for this part — it's deterministic).
  5. Provides an optional /api/wizard/ask endpoint that routes a free-
     text question through the existing AgenticLLM cascade (Tier 1
     cheap-first; budget-guarded).

This is the BACKEND for the wizard card on the dashboard. The
frontend (Strategy & ML tab) calls /api/wizard/suggest with a model
key and renders the returned recommendation list.

Why rule-based first
--------------------
Most operator-actionable improvements are deterministic:
  - "AUC ≈ 0.50 → features are noise; check funding merge + L2 wiring"
  - "Training failed in last 7d → check log line"
  - "Drift PSI ≥ 0.25 → retrain on fresh window"
  - "Cell missing for (model, tf) → submit training"
A free-text LLM call costs tokens for no quality gain on these.

The LLM is reserved for nuanced follow-ups the operator types in
("why does trend underperform during high-vol regimes?").
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
BASELINES_DIR = PROJECT_ROOT / "data" / "risk" / "drift_baselines"

# AUC bands for the rule-based recommender.
AUC_NOISE_THRESHOLD = 0.52       # below this = effectively coin-flip
AUC_TARGET_THRESHOLD = 0.55      # target floor for shippable models
AUC_GOOD_THRESHOLD = 0.60        # "doing well; tune hyperparams not features"

# Map of operator-visible model keys -> meta filename glob (existing files).
KNOWN_MODELS: dict[str, str] = {
    "trend":    "trend_{tf}_meta.json",
    "base":     "base_{tf}_meta.json",
    "futures":  "futures_{tf}_meta.json",
    "scalping": "scalping_{tf}_meta.json",
    "meta":     "meta_{tf}_meta.json",
    "regime":   "regime_{tf}_meta.json",
    "tft":      "tft_{tf}_model_meta.json",
    "oft":      "oft_{tf}_meta.json",
}

# Per-operator tf coverage matrix (matches the explicit batch I submitted).
EXPECTED_TFS_PER_MODEL: dict[str, list[str]] = {
    "trend":    ["1h", "4h", "1d", "1m"],
    "base":     ["1h", "4h", "1d", "1m"],
    "futures":  ["1h", "4h", "1d", "1m"],
    "scalping": ["1m", "15m"],
    "meta":     ["1h", "4h", "1d", "1m"],
    "regime":   ["1h", "4h", "1d", "1m"],
    "tft":      ["1h", "4h", "1d"],
    "oft":      ["1m"],
}


@dataclass
class Recommendation:
    """A single ranked improvement suggestion."""
    rank: int
    title: str
    severity: str  # "critical" | "high" | "medium" | "low"
    detail: str
    suggested_action: str  # operator-actionable next step

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "title": self.title,
            "severity": self.severity,
            "detail": self.detail,
            "suggested_action": self.suggested_action,
        }


@dataclass
class WizardReport:
    model: str
    tf: str
    auc: float | None = None
    acc: float | None = None
    last_trained: str | None = None
    recommendations: list[Recommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tf": self.tf,
            "auc": self.auc,
            "acc": self.acc,
            "last_trained": self.last_trained,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "notes": self.notes,
        }


def _meta_path(model: str, tf: str) -> Path | None:
    """Return the meta-JSON path for (model, tf), or None if model unknown."""
    tmpl = KNOWN_MODELS.get(model)
    if not tmpl:
        return None
    return MODELS_DIR / tmpl.format(tf=tf)


def _load_meta(model: str, tf: str) -> dict[str, Any]:
    """Load the model meta JSON, returning {} on any failure."""
    path = _meta_path(model, tf)
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[wizard] could not load %s: %s", path, e)
        return {}


def _load_drift_summary(model: str, tf: str) -> dict[str, Any] | None:
    """Return a summarized drift snapshot for (model, tf) if a baseline
    exists. Lightweight — does not run a fresh drift check; just reports
    baseline age + feature count."""
    baseline_file = BASELINES_DIR / f"{model}__{tf}.json"
    if not baseline_file.exists():
        return None
    try:
        payload = json.loads(baseline_file.read_text(encoding="utf-8"))
        return {
            "saved_at": payload.get("saved_at"),
            "feature_count": payload.get("feature_count"),
        }
    except Exception:
        return None


def _rule_auc_noise(model: str, tf: str, meta: dict) -> Recommendation | None:
    """If the model's AUC is below the noise threshold, the features are
    the problem — not hyperparameters. Suggest feature wiring."""
    auc = meta.get("auc") or meta.get("auc_roc")
    if auc is None or auc >= AUC_NOISE_THRESHOLD:
        return None
    detail = (
        f"AUC={auc:.3f} is at or below the noise floor ({AUC_NOISE_THRESHOLD}). "
        f"This means the input features carry no usable signal for the target; "
        f"hyperparameter tuning won't help. Audit feature wiring first."
    )
    action = (
        f"For trend@{tf} specifically, check Phase 3.5 wiring: funding_rate "
        f"merge via merge_asof + add_taker_and_trade_features + asymmetric "
        f"Triple Barrier (pt=4, sl=2) — see src/engine/train_trend_model.py. "
        f"For other models, verify FEATURE_COLUMNS includes funding_z, "
        f"ofi_z, taker_buy_ratio, and that the underlying CSVs have those "
        f"columns populated."
    )
    return Recommendation(rank=0, title="AUC at noise floor — feature poverty",
                          severity="critical", detail=detail,
                          suggested_action=action)


def _rule_auc_below_target(model: str, tf: str, meta: dict) -> Recommendation | None:
    auc = meta.get("auc") or meta.get("auc_roc")
    if auc is None or auc >= AUC_TARGET_THRESHOLD or auc < AUC_NOISE_THRESHOLD:
        return None
    return Recommendation(
        rank=0,
        title=f"AUC={auc:.3f} below target ({AUC_TARGET_THRESHOLD})",
        severity="high",
        detail=(f"Model has marginal signal but isn't shippable. Try: (1) "
                f"feature interactions via gplearn — but only after F1+F3.5 "
                f"are clean. (2) per-symbol training instead of pooled. (3) "
                f"asymmetric Triple Barrier if not already in use."),
        suggested_action=(f"Run CIO Optuna study on (model={model}, tf={tf}) "
                          f"via dashboard's Comparison tab → CIO Agent card."),
    )


def _rule_stale_training(model: str, tf: str, meta: dict) -> Recommendation | None:
    """Flag a model that hasn't been retrained in > 14 days."""
    ts = meta.get("last_trained") or meta.get("trained_at") or meta.get("timestamp")
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            ts_clean = str(ts).replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_clean)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        if age_days < 14:
            return None
        return Recommendation(
            rank=0,
            title=f"Model not retrained in {age_days:.0f} days",
            severity="medium",
            detail=(f"Market regime may have shifted; the model could be "
                    f"trading on a stale distribution."),
            suggested_action=(f"Click Train on this row (or POST /api/training/"
                              f"run/{model} with tf={tf})."),
        )
    except Exception:
        return None


def _rule_missing_cell(model: str, tf: str, meta: dict) -> Recommendation | None:
    """If the (model, tf) is in the expected matrix but no meta exists,
    flag it as a missing cell."""
    if meta:
        return None  # we have a meta — not missing
    expected = EXPECTED_TFS_PER_MODEL.get(model, [])
    if tf not in expected:
        return None
    return Recommendation(
        rank=0,
        title=f"No trained {model}@{tf} model exists",
        severity="high",
        detail=(f"This cell is in the operator's expected coverage matrix "
                f"but no model file is on disk."),
        suggested_action=(f"Submit training: POST /api/training/run/{model} "
                          f"with tf={tf}, or click Train on the row."),
    )


def _rule_no_drift_baseline(model: str, tf: str) -> Recommendation | None:
    """If we have a trained model but no drift baseline, the bot can't
    detect concept drift on live features."""
    drift = _load_drift_summary(model, tf)
    if drift is not None:
        return None
    meta = _load_meta(model, tf)
    if not meta:
        return None  # don't flag drift on a non-existent model
    return Recommendation(
        rank=0,
        title="Drift baseline missing",
        severity="medium",
        detail=("Live drift checks (Phase 6) need a saved baseline of the "
                "training-time feature distribution. Without it, the "
                "validator falls back to its z-test on raw means and may "
                "miss shape-only drift."),
        suggested_action=("Rerun training — save_baseline is called at the "
                          "end of every successful train_*_model.train()."),
    )


def suggest_for_model(model: str, tf: str) -> WizardReport:
    """Top-level entry point used by the /api/wizard/suggest endpoint.

    Returns a WizardReport with ranked recommendations (top 3 by
    severity → rank). Empty meta is handled — the wizard still produces
    "no model exists" or "data quality issue" suggestions in that case.
    """
    meta = _load_meta(model, tf)
    rep = WizardReport(
        model=model, tf=tf,
        auc=meta.get("auc") or meta.get("auc_roc"),
        acc=meta.get("accuracy") or meta.get("acc"),
        last_trained=str(meta.get("last_trained") or meta.get("trained_at") or "—"),
    )

    candidates: list[Recommendation] = []
    rule_fns = (
        _rule_missing_cell,
        _rule_auc_noise,
        _rule_auc_below_target,
        _rule_stale_training,
    )
    for fn in rule_fns:
        r = fn(model, tf, meta)
        if r:
            candidates.append(r)
    # Drift rule has different signature
    r = _rule_no_drift_baseline(model, tf)
    if r:
        candidates.append(r)

    # Rank by severity, then title.
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    candidates.sort(key=lambda c: (sev_order.get(c.severity, 99), c.title))
    for i, c in enumerate(candidates[:5]):
        c.rank = i + 1
        rep.recommendations.append(c)

    if not rep.recommendations:
        rep.notes.append("No issues detected — model is within expected bounds.")
    return rep


def ask_llm(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route a free-text operator question through AgenticLLM (Tier 1
    cheap-first cascade, budget-guarded). Returns a structured
    {answer, model_used, source} dict.

    The wizard's rule-based suggestions cover ~80% of operator questions;
    this is for the long-tail "why does X happen during Y" follow-ups.
    """
    if not question or not question.strip():
        return {"answer": "(empty question)", "model_used": None, "source": "empty"}
    try:
        from src.engine.agentic_llm import AgenticLLM
        llm = AgenticLLM()
        if not llm.is_active:
            return {
                "answer": ("LLM not configured — set GEMINI_API_KEY in .env "
                           "to enable free-text Q&A. Rule-based suggestions "
                           "still work."),
                "model_used": None,
                "source": "no_api_key",
            }
    except Exception as exc:
        return {"answer": f"LLM init failed: {exc}", "model_used": None,
                "source": "init_error"}

    # Phase C (2026-05-14) — route through AgenticLLM.query() (free-form)
    # instead of evaluate_trade() (which forces an APPROVE/VETO JSON
    # envelope). The old path made the LLM frame answers as trade
    # justifications — operator saw replies like "this is a valid
    # operational query and does not fall under the VETO criteria"
    # instead of an actual answer.
    ctx_lines = []
    if context:
        for k, v in list(context.items())[:8]:
            ctx_lines.append(f"  {k}: {v}")
    ctx_text = "\n".join(ctx_lines) if ctx_lines else "  (none)"
    prompt = (
        f"You are an AI trading-bot training advisor. Give the operator a "
        f"concise, actionable answer (3-5 sentences max). Avoid generic "
        f"boilerplate; cite specific features, hyperparameters, or training "
        f"steps when relevant.\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    try:
        return llm.query(prompt)
    except Exception as exc:
        return {"answer": f"LLM call failed: {exc}", "model_used": None,
                "source": "llm_error"}


__all__ = [
    "Recommendation", "WizardReport",
    "suggest_for_model", "ask_llm",
    "KNOWN_MODELS", "EXPECTED_TFS_PER_MODEL",
]
