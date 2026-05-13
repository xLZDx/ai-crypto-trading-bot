"""
ML Engineer Agent — Quantitative Finance Training Pipeline Gatekeeper.

This agent is the system-level enforcement of the AFML (Lopez de Prado) +
BryceMeng/mlfinlab methodology. It runs on EVERY training task:

  Pre-flight gate (before dispatch to cluster)
    1. Validates the training config against AFML hard rules
    2. Checks META_FEATURES integrity if training a meta-labeler
    3. Verifies prerequisite data + primary models exist
    4. Confirms ATR / FracDiff d-value freshness
    5. Returns APPROVE / BLOCK / WARN with reasons

  Post-training gate (after model artifact written)
    1. Loads the freshly trained model meta JSON
    2. Validates feature count matches META_FEATURES list
    3. Checks walk-forward Sharpe / Sortino / accuracy against floors
    4. Enforces PSR (Probabilistic Sharpe Ratio) for multi-test bias
    5. Returns ACCEPT / REJECT / FLAG_FOR_REVIEW

Decisions persisted to: data/ml_engineer_decisions.json

This agent is the AUTHORITATIVE source for "is this model ready for production".
The cluster orchestrator calls it via `validate_training_request()` before any
worker receives a task, and via `evaluate_trained_model()` before any model is
promoted to the live inference engine.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.meta_config import META_FEATURES, CONFIDENCE_THRESHOLD
from src.utils.safe_json import write_json, read_json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECISIONS_PATH = PROJECT_ROOT / 'data' / 'ml_engineer_decisions.json'


# ── AFML hard rules ──────────────────────────────────────────────────────────

AFML_RULES: dict[str, Any] = {
    'triple_barrier': {
        'pt_multiplier_min':  2.0,
        'pt_multiplier_max':  3.0,
        'sl_multiplier_min':  1.0,
        'sl_multiplier_max':  2.0,
        'max_bars_min':       6,
        'max_bars_max':       24,
        'asymmetric_ratio_min': 1.3,   # pt/sl must be ≥ 1.3 (positive expectancy)
        'preferred_pt': 2.5,
        'preferred_sl': 1.5,
        'preferred_max_bars': 12,
    },
    'purged_kfold': {
        'min_pct_embargo': 0.0,        # pct_embargo=0 is allowed but must use t1 purging
        'max_pct_embargo': 0.20,
        'require_t1':      True,       # PurgedKFold without t1 is NOT purged
    },
    'meta_labeler': {
        'n_features_expected': len(META_FEATURES),
        'min_train_samples':   500,
        'min_win_rate_pct':    35.0,   # < 35% = mislabeled or broken pipeline
        'max_win_rate_pct':    75.0,   # > 75% = look-ahead bias suspected
        'min_walk_forward_acc': 50.0,  # WF accuracy must beat coin-flip
        'min_test_acc':        52.0,
        'min_auc':             0.55,
    },
    'fractional_diff': {
        'd_min':               0.30,
        'd_max':               0.50,
        'adf_pvalue_threshold': 0.05,  # ADF must reject unit root at 5%
        'min_correlation':     0.90,   # frac_diff vs original must keep ≥ 90% correlation
    },
    'training_pipeline': {
        'calibration_method':  'isotonic',
        'class_weight':        'balanced',
        'require_optimal_threshold': True,
    },
}


@dataclass
class MLEngineerDecision:
    """Single audit record for a pre- or post-flight gate."""
    timestamp: str
    phase: str                      # 'pre_flight' | 'post_training'
    model_type: str                 # 'meta' | 'base' | 'trend' | 'scalping' | 'tft' | 'oft' | 'regime'
    timeframe: str
    decision: str                   # 'APPROVE' | 'BLOCK' | 'WARN' | 'ACCEPT' | 'REJECT' | 'FLAG_FOR_REVIEW'
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    config:  dict[str, Any] = field(default_factory=dict)


class MLEngineerAgent:
    """
    Stateless validator. Reads AFML_RULES at every call so changes take effect
    without restarting the orchestrator.
    """

    def __init__(self, decisions_path: Path | None = None):
        self.decisions_path = decisions_path or DECISIONS_PATH
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight: validate before dispatch to cluster ──────────────────────

    def validate_training_request(
        self,
        model_type: str,
        timeframe: str,
        config: dict[str, Any] | None = None,
    ) -> MLEngineerDecision:
        """
        Run AFML pre-flight checks. Called by cluster_orchestrator.submit_task()
        before the task is queued. BLOCK = task is refused.
        """
        cfg = config or {}
        decision = MLEngineerDecision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase='pre_flight',
            model_type=model_type,
            timeframe=timeframe,
            decision='APPROVE',  # tentative
            config=cfg,
        )

        # ── Triple Barrier checks (apply to all models that label trades) ──
        if model_type in ('meta', 'base', 'trend', 'scalping', 'futures'):
            self._check_triple_barrier(cfg, decision)
            self._check_purged_kfold(cfg, decision)

        # ── META_FEATURES integrity (meta-labeler specific) ──
        if model_type == 'meta':
            self._check_meta_features_unified(decision)
            self._check_primary_models_exist(decision)

        # ── Training pipeline rules ──
        self._check_pipeline_config(cfg, decision)

        # ── Sprint 0 §S0-1 Validation rigor (data freshness + label balance) ──
        # Runs when `validate_data=True` in config (default off — opt-in until
        # the trainer wires it consistently). Stale-data BLOCKs propagate.
        if cfg.get('validate_data'):
            try:
                from src.risk.validators import get_validation_gate
                vreport = get_validation_gate().run(
                    model_type=model_type,
                    timeframe=timeframe,
                )
                if vreport.decision == 'BLOCK':
                    decision.reasons.extend(vreport.reasons)
                elif vreport.decision == 'WARN':
                    decision.warnings.extend(vreport.warnings)
                decision.metrics.setdefault('validation', {})
                decision.metrics['validation'] = vreport.metrics
            except Exception as e:
                decision.warnings.append(
                    f'WARN: ValidationGate unavailable ({e}); proceeding without data checks'
                )

        # Resolve final decision
        if any(r.startswith('BLOCK:') for r in decision.reasons):
            decision.decision = 'BLOCK'
        elif decision.warnings:
            decision.decision = 'WARN'
        else:
            decision.decision = 'APPROVE'

        self._persist_decision(decision)
        self._log_decision(decision)
        return decision

    # ── Post-training: validate after artifact written ───────────────────────

    def evaluate_trained_model(
        self,
        model_type: str,
        timeframe: str,
        meta_json_path: Path | str,
    ) -> MLEngineerDecision:
        """
        Validate a trained model against KPI floors. Called after the worker
        finishes writing the artifact + meta JSON. REJECT = artifact is
        quarantined and NOT promoted to live inference.
        """
        meta_path = Path(meta_json_path)
        decision = MLEngineerDecision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase='post_training',
            model_type=model_type,
            timeframe=timeframe,
            decision='ACCEPT',  # tentative
            config={'meta_json_path': str(meta_path)},
        )

        if not meta_path.exists():
            decision.reasons.append(f'BLOCK: meta JSON not found at {meta_path}')
            decision.decision = 'REJECT'
            self._persist_decision(decision)
            return decision

        try:
            with open(meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except Exception as e:
            decision.reasons.append(f'BLOCK: could not parse meta JSON: {e}')
            decision.decision = 'REJECT'
            self._persist_decision(decision)
            return decision

        decision.metrics = {
            'accuracy': meta.get('accuracy'),
            'auc_roc':  meta.get('auc_roc'),
            'walk_forward_mean_acc': meta.get('walk_forward_mean_acc'),
            'walk_forward_std_acc':  meta.get('walk_forward_std_acc'),
            'win_precision': meta.get('win_precision'),
            'win_rate_pct':  meta.get('win_rate_pct'),
            'optimal_threshold': meta.get('optimal_threshold') or meta.get('confidence_threshold'),
            'optimal_sortino':   meta.get('optimal_sortino'),
            'n_features':    meta.get('n_features'),
            'n_train':       meta.get('n_train'),
            'n_test':        meta.get('n_test'),
        }

        # ── Apply model-type-specific KPI floors ──
        if model_type == 'meta':
            self._check_meta_kpi(meta, decision)
        else:
            self._check_general_kpi(meta, decision)

        # ── PSR (Probabilistic Sharpe Ratio) deflation check ──
        # Per Bailey & Lopez de Prado: PSR needs the actual observed Sharpe and
        # the number of return observations. The previous code used WF accuracy
        # percentage as the Sharpe input, which saturated PSR at 1.0 for every
        # trained model (dead-code guard). We now pull a real OOS Sharpe from
        # the meta JSON if available; if the trainer hasn't written one, we
        # emit an info-level warning that PSR was not evaluated and skip the
        # gate (rather than silently passing with a fake value).
        oos_sharpe = meta.get('oos_sharpe') or meta.get('oos_sharpe_ratio')
        n_obs = meta.get('n_test') or meta.get('n_oos_returns')
        if oos_sharpe is not None and n_obs and float(n_obs) > 1:
            psr = self._compute_psr(
                observed_sr=float(oos_sharpe),
                n_obs=int(n_obs),
                sr_benchmark=0.0,
                skew=float(meta.get('oos_return_skew') or 0.0),
                kurtosis=float(meta.get('oos_return_kurtosis') or 3.0),
            )
            decision.metrics['psr'] = round(psr, 4)
            if psr < 0.6:
                decision.warnings.append(
                    f'WARN: low PSR ({psr:.2f}) — observed Sharpe ({oos_sharpe:.3f}) '
                    f'on {n_obs} OOS returns is statistically thin. Apply Deflated '
                    'Sharpe before promoting to live.'
                )
        else:
            decision.warnings.append(
                'WARN: PSR not evaluated — trainer did not write `oos_sharpe` + '
                '`n_test`/`n_oos_returns`. Add them to enable multi-test bias guard.'
            )

        # ── Resolve final decision ──
        if any(r.startswith('BLOCK:') for r in decision.reasons):
            decision.decision = 'REJECT'
        elif decision.warnings:
            decision.decision = 'FLAG_FOR_REVIEW'
        else:
            decision.decision = 'ACCEPT'

        self._persist_decision(decision)
        self._log_decision(decision)
        return decision

    # ── Internal checks ──────────────────────────────────────────────────────

    def _check_triple_barrier(self, cfg: dict, decision: MLEngineerDecision) -> None:
        """AFML Ch.3 Triple Barrier compliance."""
        rules = AFML_RULES['triple_barrier']
        pt = cfg.get('pt_multiplier')
        sl = cfg.get('sl_multiplier')
        mb = cfg.get('max_bars')

        # If caller didn't specify, the code defaults are honored — but warn.
        if pt is None or sl is None or mb is None:
            decision.warnings.append(
                'WARN: triple_barrier params not specified — code defaults '
                f'(pt={rules["preferred_pt"]}, sl={rules["preferred_sl"]}, '
                f'max_bars={rules["preferred_max_bars"]}) will be used.'
            )
            return

        if not (rules['pt_multiplier_min'] <= pt <= rules['pt_multiplier_max']):
            decision.reasons.append(
                f'BLOCK: pt_multiplier={pt} outside AFML range '
                f'[{rules["pt_multiplier_min"]}, {rules["pt_multiplier_max"]}]'
            )
        if not (rules['sl_multiplier_min'] <= sl <= rules['sl_multiplier_max']):
            decision.reasons.append(
                f'BLOCK: sl_multiplier={sl} outside AFML range '
                f'[{rules["sl_multiplier_min"]}, {rules["sl_multiplier_max"]}]'
            )
        if not (rules['max_bars_min'] <= mb <= rules['max_bars_max']):
            decision.reasons.append(
                f'BLOCK: max_bars={mb} outside AFML range '
                f'[{rules["max_bars_min"]}, {rules["max_bars_max"]}]'
            )
        if sl > 0 and (pt / sl) < rules['asymmetric_ratio_min']:
            decision.reasons.append(
                f'BLOCK: pt/sl ratio={pt/sl:.2f} below AFML minimum '
                f'{rules["asymmetric_ratio_min"]} — barriers are too symmetric, '
                'positive expectancy is not guaranteed.'
            )

    def _check_purged_kfold(self, cfg: dict, decision: MLEngineerDecision) -> None:
        rules = AFML_RULES['purged_kfold']
        pct_embargo = cfg.get('pct_embargo')
        # Review fix: previous default `use_t1_purging=True` meant ANY empty
        # config silently satisfied the gate. The caller now must EXPLICITLY
        # opt-in (use_t1_purging=True or t1_series present) — empty config
        # → BLOCK with a clear reason.
        t1_provided = bool(cfg.get('t1_series') or cfg.get('use_t1_purging'))

        if pct_embargo is not None and not (rules['min_pct_embargo'] <= pct_embargo <= rules['max_pct_embargo']):
            decision.warnings.append(
                f'WARN: pct_embargo={pct_embargo} outside recommended '
                f'[{rules["min_pct_embargo"]}, {rules["max_pct_embargo"]}].'
            )
        if rules['require_t1'] and not t1_provided:
            decision.reasons.append(
                'BLOCK: t1 series must be provided to PurgedKFold for AFML '
                'label-span purging. Embargo alone is insufficient. '
                'Pass `use_t1_purging=True` or `t1_series` in the config.'
            )

    def _check_meta_features_unified(self, decision: MLEngineerDecision) -> None:
        """Verify training and inference share the same META_FEATURES list."""
        try:
            # Both files now import from src.utils.meta_config — verify by import.
            from src.utils.meta_config import META_FEATURES as canonical
            from src.engine.train_meta_labeler import META_FEATURES as training
            from src.analysis.meta_labeler import META_FEATURES as inference
            if list(canonical) != list(training):
                decision.reasons.append(
                    'BLOCK: train_meta_labeler META_FEATURES diverged from '
                    'src.utils.meta_config — re-import.'
                )
            if list(canonical) != list(inference):
                decision.reasons.append(
                    'BLOCK: meta_labeler (inference) META_FEATURES diverged '
                    'from src.utils.meta_config — re-import.'
                )
        except ImportError as e:
            # ImportError = structural breakage (file deleted/renamed/circular),
            # not a transient runtime issue. BLOCK rather than WARN — this is
            # exactly the configuration drift the gate exists to catch.
            decision.reasons.append(
                f'BLOCK: META_FEATURES import failed ({e}). The training or '
                'inference module is missing/broken — fix before training.'
            )
        except Exception as e:
            decision.warnings.append(
                f'WARN: could not verify META_FEATURES unification: {e}'
            )

    def _check_primary_models_exist(self, decision: MLEngineerDecision) -> None:
        """Meta-labeler training requires base + trend + regime models."""
        models_dir = PROJECT_ROOT / 'models'
        required = ['btc_rf_model.joblib', 'trend_model.joblib', 'regime_classifier.joblib']
        for name in required:
            p = models_dir / name
            if not p.exists():
                decision.reasons.append(
                    f'BLOCK: meta-labeler requires {name} but it is missing. '
                    'Train primary models first.'
                )

    def _check_pipeline_config(self, cfg: dict, decision: MLEngineerDecision) -> None:
        rules = AFML_RULES['training_pipeline']
        calib = cfg.get('calibration_method')
        if calib is not None and calib != rules['calibration_method']:
            decision.warnings.append(
                f'WARN: calibration_method={calib} — AFML recommends '
                f'{rules["calibration_method"]} for binary classifiers.'
            )
        cw = cfg.get('class_weight')
        if cw is not None and cw != rules['class_weight']:
            decision.warnings.append(
                f'WARN: class_weight={cw} — recommended {rules["class_weight"]} '
                'for imbalanced trade outcomes.'
            )

    def _check_meta_kpi(self, meta: dict, decision: MLEngineerDecision) -> None:
        rules = AFML_RULES['meta_labeler']
        n_feat = meta.get('n_features')
        if n_feat is not None and n_feat != rules['n_features_expected']:
            decision.reasons.append(
                f'BLOCK: meta-labeler n_features={n_feat} expected '
                f'{rules["n_features_expected"]} (META_FEATURES drift).'
            )
        n_train = meta.get('n_train', 0) or 0
        if n_train < rules['min_train_samples']:
            decision.reasons.append(
                f'BLOCK: only {n_train} training samples (< {rules["min_train_samples"]}).'
            )
        win_rate = float(meta.get('win_rate_pct') or 0.0)
        if win_rate < rules['min_win_rate_pct']:
            decision.warnings.append(
                f'WARN: win_rate={win_rate:.1f}% is suspiciously low — '
                'possible mislabeling or broken Triple Barrier.'
            )
        if win_rate > rules['max_win_rate_pct']:
            decision.warnings.append(
                f'WARN: win_rate={win_rate:.1f}% is suspiciously high — '
                'check for look-ahead bias.'
            )
        wf_acc = float(meta.get('walk_forward_mean_acc') or 0.0)
        if wf_acc < rules['min_walk_forward_acc']:
            decision.reasons.append(
                f'BLOCK: walk-forward accuracy={wf_acc:.1f}% does not beat '
                f'coin-flip ({rules["min_walk_forward_acc"]}%).'
            )
        test_acc = float(meta.get('accuracy') or 0.0)
        if test_acc < rules['min_test_acc']:
            decision.warnings.append(
                f'WARN: test accuracy={test_acc:.1f}% below floor '
                f'{rules["min_test_acc"]}%.'
            )
        auc = float(meta.get('auc_roc') or 0.5)
        if auc < rules['min_auc']:
            decision.warnings.append(
                f'WARN: AUC={auc:.3f} below floor {rules["min_auc"]} — model '
                'barely distinguishes wins from losses.'
            )
        opt_thr = meta.get('optimal_threshold')
        if opt_thr is None and AFML_RULES['training_pipeline']['require_optimal_threshold']:
            decision.reasons.append(
                'BLOCK: optimal_threshold not persisted in meta JSON — '
                'inference will fall back to hardcoded 0.60.'
            )

    def _check_general_kpi(self, meta: dict, decision: MLEngineerDecision) -> None:
        """KPI floors for non-meta models (base, trend, scalping, etc.)."""
        wf_acc = float(meta.get('walk_forward_mean_acc') or meta.get('accuracy') or 0.0)
        if wf_acc < 50.0:
            decision.warnings.append(
                f'WARN: model accuracy={wf_acc:.1f}% does not beat coin-flip.'
            )

    @staticmethod
    def _compute_psr(observed_sr: float, n_obs: int,
                     sr_benchmark: float = 0.0,
                     skew: float = 0.0, kurtosis: float = 3.0) -> float:
        """
        Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado (2012), Eq. 4.

        Returns the probability that the TRUE Sharpe ratio exceeds `sr_benchmark`
        given an observed Sharpe `observed_sr` from `n_obs` returns.

        The denominator accounts for the skew/kurtosis of returns. Under IID
        normality (skew=0, kurtosis=3), this reduces to:
            denom = sqrt(1 - 0 * SR + (3-1)/4 * SR²) = sqrt(1 + SR²/2)

        Args:
            observed_sr:  Observed Sharpe ratio (e.g. mean/std of OOS returns).
                          Must be in the same time-unit as n_obs (e.g. daily SR
                          with n_obs = number of trading days).
            n_obs:        Number of return observations (NOT number of folds).
            sr_benchmark: Target Sharpe to test against (default 0).
            skew:         Sample skew of returns (default 0 = IID normal).
            kurtosis:     Sample kurtosis of returns (default 3 = IID normal).
        """
        if n_obs <= 1:
            return 0.0
        # Bailey-LdP denominator under generic moments
        denom_sq = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
        if denom_sq <= 0:
            return 0.0
        denom = math.sqrt(denom_sq)
        from math import erf
        z = (observed_sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
        # Standard normal CDF
        return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist_decision(self, decision: MLEngineerDecision) -> None:
        """Append decision to data/ml_engineer_decisions.json (atomic via safe_json)."""
        try:
            existing = read_json(str(self.decisions_path), default={'decisions': []})
            if not isinstance(existing, dict):
                existing = {'decisions': []}
            decisions_list = existing.get('decisions') or []
            decisions_list.append(asdict(decision))
            # Keep last 500 decisions for size control
            existing['decisions'] = decisions_list[-500:]
            write_json(str(self.decisions_path), existing)
        except Exception as e:
            logger.error("ML Engineer agent: failed to persist decision: %s", e)

    def _log_decision(self, decision: MLEngineerDecision) -> None:
        level = logging.INFO
        if decision.decision in ('BLOCK', 'REJECT'):
            level = logging.ERROR
        elif decision.decision in ('WARN', 'FLAG_FOR_REVIEW'):
            level = logging.WARNING
        logger.log(
            level,
            "[ML Engineer] %s/%s @ %s → %s | reasons=%s | warnings=%s",
            decision.model_type, decision.timeframe, decision.phase,
            decision.decision, decision.reasons, decision.warnings,
        )


# ── Module-level convenience singleton ───────────────────────────────────────

_agent_singleton: MLEngineerAgent | None = None


def get_ml_engineer() -> MLEngineerAgent:
    """Lazy singleton — call from cluster orchestrator."""
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = MLEngineerAgent()
    return _agent_singleton
