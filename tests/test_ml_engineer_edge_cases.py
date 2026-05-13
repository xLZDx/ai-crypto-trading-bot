"""
ML Engineer Agent — Edge-Case Behavioral Tests.

Tests beyond what exists in test_phase0_fix.py. Every test here:
  1. Calls the function under test directly.
  2. Asserts on observable behavior (return value, side-effects, state).
  3. Never uses string-match as the only assertion.

Coverage targets (see docstring on each group):
  - Pre-flight gate: no config, boundary values, zero SL, unknown model_type
  - Post-training gate: missing file, malformed JSON, feature count drift,
    floor-inclusive acceptance, high/low win_rate flags, PSR edge cases
  - Persistence: read-only dir, 500-entry cap
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _good_meta(overrides: dict | None = None) -> dict:
    """Return a fully-valid meta JSON that passes all KPI floors."""
    base = {
        'accuracy': 55.0,
        'auc_roc': 0.60,
        'win_precision': 55.0,
        'win_rate_pct': 50.0,
        'walk_forward_mean_acc': 53.0,
        'walk_forward_std_acc': 2.0,
        'walk_forward_folds': 5,
        'optimal_threshold': 0.55,
        'n_features': 23,        # matches META_FEATURES length
        'n_train': 1000,
        'n_test': 300,
    }
    if overrides:
        base.update(overrides)
    return base


def _agent(tmp_path: Path):
    from src.engine.ml_engineer_agent import MLEngineerAgent
    return MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')


# ── Group 1: Pre-flight — no config (defaults path) ──────────────────────────

class TestPreFlightNoConfig:
    """Edge 1: no config dict passed → WARN about defaults, not BLOCK."""

    def test_block_with_no_config_meta_model(self, tmp_path):
        """Post-review fix: empty config now BLOCKs (use_t1_purging no longer
        defaults to True). Caller must EXPLICITLY opt in to AFML purging."""
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='meta', timeframe='1h', config=None
        )
        assert decision.decision == 'BLOCK', (
            f"Expected BLOCK with no config (missing t1 opt-in), "
            f"got {decision.decision}. Reasons: {decision.reasons}"
        )
        assert any('t1' in r.lower() for r in decision.reasons)

    def test_block_with_no_config_base_model(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='15m', config=None
        )
        assert decision.decision == 'BLOCK'
        assert any('t1' in r.lower() for r in decision.reasons)

    def test_approve_with_explicit_t1_opt_in(self, tmp_path):
        """When the caller EXPLICITLY opts in, AFML-compliant config → APPROVE."""
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'max_bars': 12, 'use_t1_purging': True},
        )
        assert decision.decision == 'APPROVE'
        assert not any(r.startswith('BLOCK:') for r in decision.reasons)


# ── Group 2: Pre-flight — preferred values (golden path) ─────────────────────

class TestPreFlightPreferredValues:
    """Edge 2: pt=2.5, sl=1.5 (AFML preferred) → APPROVE, no warnings about barriers."""

    def test_preferred_values_approve_no_barrier_warnings(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5,
                'sl_multiplier': 1.5,
                'max_bars': 12,
                'use_t1_purging': True,
            },
        )
        assert decision.decision == 'APPROVE', (
            f"Preferred AFML values should APPROVE, got {decision.decision}: {decision.reasons}"
        )
        assert not any(r.startswith('BLOCK:') for r in decision.reasons)
        # No barrier-specific warnings
        barrier_warnings = [w for w in decision.warnings if 'pt_multiplier' in w or 'sl_multiplier' in w]
        assert barrier_warnings == [], f"Unexpected barrier warnings: {barrier_warnings}"


# ── Group 3: Pre-flight — exact boundary values ───────────────────────────────

class TestPreFlightBoundaryValues:
    """Edges 3 & 4: pt at min (2.0) and max (3.0) boundaries → APPROVE."""

    def test_pt_multiplier_at_min_boundary_approved(self, tmp_path):
        agent = _agent(tmp_path)
        # pt=2.0, sl=1.0 → ratio=2.0 ≥ 1.3 → should APPROVE
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.0,
                'sl_multiplier': 1.0,
                'max_bars': 12,
                'use_t1_purging': True,
            },
        )
        assert not any(r.startswith('BLOCK:') and 'pt_multiplier' in r for r in decision.reasons), (
            f"pt_multiplier=2.0 is at min boundary and must not BLOCK: {decision.reasons}"
        )

    def test_pt_multiplier_at_max_boundary_approved(self, tmp_path):
        agent = _agent(tmp_path)
        # pt=3.0, sl=1.5 → ratio=2.0 ≥ 1.3 → APPROVE
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 3.0,
                'sl_multiplier': 1.5,
                'max_bars': 12,
                'use_t1_purging': True,
            },
        )
        assert not any(r.startswith('BLOCK:') and 'pt_multiplier' in r for r in decision.reasons), (
            f"pt_multiplier=3.0 is at max boundary and must not BLOCK: {decision.reasons}"
        )

    def test_max_bars_at_min_boundary_approved(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5,
                'sl_multiplier': 1.5,
                'max_bars': 6,   # min boundary
                'use_t1_purging': True,
            },
        )
        assert not any(r.startswith('BLOCK:') and 'max_bars' in r for r in decision.reasons)

    def test_max_bars_at_max_boundary_approved(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5,
                'sl_multiplier': 1.5,
                'max_bars': 24,  # max boundary
                'use_t1_purging': True,
            },
        )
        assert not any(r.startswith('BLOCK:') and 'max_bars' in r for r in decision.reasons)


# ── Group 4: Pre-flight — sl_multiplier=0 (divide-by-zero guard) ─────────────

class TestPreFlightZeroSL:
    """Edge 5: sl_multiplier=0 → BLOCK (zero division in ratio check)."""

    def test_sl_zero_is_blocked(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5,
                'sl_multiplier': 0.0,   # zero → outside [1.0, 2.0]
                'max_bars': 12,
                'use_t1_purging': True,
            },
        )
        assert decision.decision == 'BLOCK', (
            "sl_multiplier=0 is outside AFML range and must be BLOCKED"
        )
        assert any('sl_multiplier' in r for r in decision.reasons)

    def test_sl_zero_does_not_raise_exception(self, tmp_path):
        """The agent must NEVER crash on sl=0, even if the ratio check tried pt/0."""
        agent = _agent(tmp_path)
        try:
            decision = agent.validate_training_request(
                model_type='base', timeframe='1h',
                config={
                    'pt_multiplier': 2.5,
                    'sl_multiplier': 0.0,
                    'max_bars': 12,
                    'use_t1_purging': True,
                },
            )
        except ZeroDivisionError as exc:
            pytest.fail(f"Agent crashed with ZeroDivisionError on sl=0: {exc}")
        # The decision object must exist and be BLOCK
        assert decision is not None


# ── Group 5: Pre-flight — unknown model_type ──────────────────────────────────

class TestPreFlightUnknownModelType:
    """Edge 7: model_type not in known list → no crash, returns APPROVE (no AFML checks apply)."""

    def test_unknown_model_type_does_not_crash(self, tmp_path):
        agent = _agent(tmp_path)
        # 'unknown' is not in ('meta','base','trend','scalping','futures')
        decision = agent.validate_training_request(
            model_type='unknown', timeframe='1h', config=None
        )
        assert decision is not None
        assert decision.model_type == 'unknown'

    def test_unknown_model_type_approve_no_afml_blocks(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='unknown', timeframe='1h',
            config={
                'pt_multiplier': 0.1,   # would BLOCK for known types
                'sl_multiplier': 0.0,
                'max_bars': 1,
            },
        )
        # Triple-barrier check only runs for known model types; unknown skips it
        barrier_blocks = [r for r in decision.reasons if 'pt_multiplier' in r or 'sl_multiplier' in r or 'max_bars' in r]
        assert barrier_blocks == [], (
            f"Unknown model_type should not apply triple-barrier AFML checks: {barrier_blocks}"
        )


# ── Group 6: Post-training — missing meta JSON ────────────────────────────────

class TestPostTrainingMissingFile:
    """Edge 8: meta JSON file does not exist → REJECT."""

    def test_missing_meta_json_is_rejected(self, tmp_path):
        agent = _agent(tmp_path)
        missing_path = tmp_path / 'nonexistent_model_meta.json'
        decision = agent.evaluate_trained_model('meta', '1h', missing_path)
        assert decision.decision == 'REJECT', (
            f"Missing meta JSON must produce REJECT, got {decision.decision}"
        )
        assert any('not found' in r.lower() for r in decision.reasons)

    def test_missing_meta_json_does_not_crash(self, tmp_path):
        agent = _agent(tmp_path)
        try:
            decision = agent.evaluate_trained_model('meta', '1h', tmp_path / 'ghost.json')
        except Exception as exc:
            pytest.fail(f"Agent crashed on missing meta JSON: {exc}")
        assert decision is not None


# ── Group 7: Post-training — malformed JSON ───────────────────────────────────

class TestPostTrainingMalformedJSON:
    """Edge 9: meta JSON exists but contains invalid JSON → REJECT, no crash."""

    def test_malformed_json_is_rejected(self, tmp_path):
        bad_file = tmp_path / 'bad_meta.json'
        bad_file.write_text("{ this is NOT valid JSON !!!", encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', bad_file)
        assert decision.decision == 'REJECT'
        assert any('parse' in r.lower() or 'json' in r.lower() for r in decision.reasons)

    def test_malformed_json_does_not_raise(self, tmp_path):
        bad_file = tmp_path / 'bad_meta.json'
        bad_file.write_text("null\x00binary\xff\xfe garbage", encoding='latin-1')
        agent = _agent(tmp_path)
        try:
            decision = agent.evaluate_trained_model('meta', '1h', bad_file)
        except Exception as exc:
            pytest.fail(f"Agent crashed on malformed meta JSON: {exc}")
        assert decision is not None
        assert decision.decision == 'REJECT'


# ── Group 8: Post-training — feature count drift ─────────────────────────────

class TestPostTrainingFeatureCountDrift:
    """Edge 10: n_features=22 (off-by-one from expected 23) → BLOCK."""

    def test_n_features_off_by_one_below_blocks(self, tmp_path):
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(_good_meta({'n_features': 22})), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        assert decision.decision == 'REJECT', (
            f"n_features=22 vs expected 23 must REJECT, got {decision.decision}"
        )
        assert any('n_features' in r for r in decision.reasons)

    def test_n_features_correct_does_not_block(self, tmp_path):
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(_good_meta({'n_features': 23})), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        assert not any('n_features' in r for r in decision.reasons)

    def test_n_features_above_blocks(self, tmp_path):
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(_good_meta({'n_features': 24})), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        assert decision.decision == 'REJECT'
        assert any('n_features' in r for r in decision.reasons)


# ── Group 9: Post-training — floor-inclusive ACCEPT ──────────────────────────

class TestPostTrainingFloorInclusive:
    """Edge 11: all KPIs at exactly floor values → ACCEPT (floors are inclusive)."""

    def test_kpis_at_exact_floor_accepted(self, tmp_path):
        # Use the minimum-passing values for every field
        meta = {
            'accuracy': 52.0,              # min_test_acc=52.0 floor
            'auc_roc': 0.55,               # min_auc=0.55 floor
            'win_precision': 50.0,
            'win_rate_pct': 35.0,          # min_win_rate_pct=35.0 floor
            'walk_forward_mean_acc': 50.0, # min_walk_forward_acc=50.0 floor
            'walk_forward_std_acc': 0.0,   # std=0 → PSR path bypassed
            'walk_forward_folds': 5,
            'optimal_threshold': 0.55,
            'n_features': 23,
            'n_train': 500,                # min_train_samples=500 floor
            'n_test': 100,
        }
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        # All floors inclusive → no BLOCK reasons expected
        block_reasons = [r for r in decision.reasons if r.startswith('BLOCK:')]
        assert block_reasons == [], (
            f"KPIs at floor values should NOT trigger BLOCKs: {block_reasons}"
        )
        assert decision.decision in ('ACCEPT', 'FLAG_FOR_REVIEW'), (
            f"Floor-inclusive meta should ACCEPT or FLAG_FOR_REVIEW, got {decision.decision}"
        )


# ── Group 10: Post-training — win_rate extremes → FLAG_FOR_REVIEW ─────────────

class TestPostTrainingWinRateFlags:
    """Edges 12 & 13: win_rate >75% or <35% → FLAG_FOR_REVIEW (look-ahead / mislabel)."""

    def test_very_high_win_rate_flagged(self, tmp_path):
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(
            json.dumps(_good_meta({'win_rate_pct': 80.0})), encoding='utf-8'
        )
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        assert decision.decision == 'FLAG_FOR_REVIEW', (
            f"win_rate=80% should FLAG_FOR_REVIEW, got {decision.decision}"
        )
        assert any('look-ahead' in w.lower() or 'high' in w.lower() for w in decision.warnings)

    def test_very_low_win_rate_flagged(self, tmp_path):
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(
            json.dumps(_good_meta({'win_rate_pct': 20.0})), encoding='utf-8'
        )
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        assert decision.decision == 'FLAG_FOR_REVIEW', (
            f"win_rate=20% should FLAG_FOR_REVIEW, got {decision.decision}"
        )
        assert any('mislabel' in w.lower() or 'low' in w.lower() for w in decision.warnings)

    def test_win_rate_exactly_at_max_boundary_not_flagged(self, tmp_path):
        """win_rate=75.0 is at the exclusive boundary — only >75.0 triggers the flag."""
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(
            json.dumps(_good_meta({'win_rate_pct': 75.0})), encoding='utf-8'
        )
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        high_win_warnings = [w for w in decision.warnings if 'look-ahead' in w.lower()]
        assert high_win_warnings == [], (
            f"win_rate=75.0 is AT the boundary and must NOT trigger the look-ahead flag: {high_win_warnings}"
        )

    def test_win_rate_exactly_at_min_boundary_not_flagged(self, tmp_path):
        """win_rate=35.0 is at the inclusive floor — not below it."""
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(
            json.dumps(_good_meta({'win_rate_pct': 35.0})), encoding='utf-8'
        )
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        low_win_warnings = [w for w in decision.warnings if 'mislabel' in w.lower()]
        assert low_win_warnings == [], (
            f"win_rate=35.0 is AT the floor and must NOT trigger the mislabel flag: {low_win_warnings}"
        )


# ── Group 11: PSR computation edge cases ─────────────────────────────────────

class TestPSRComputation:
    """Edges 14 & 15: PSR math edge cases."""

    def test_psr_one_obs_returns_zero(self):
        """Post-review: Bailey-LdP PSR signature now (observed_sr, n_obs, ...)."""
        from src.engine.ml_engineer_agent import MLEngineerAgent
        result = MLEngineerAgent._compute_psr(observed_sr=2.0, n_obs=1)
        assert result == 0.0, f"n_obs=1 must return 0.0, got {result}"

    def test_psr_zero_obs_returns_zero(self):
        from src.engine.ml_engineer_agent import MLEngineerAgent
        result = MLEngineerAgent._compute_psr(observed_sr=2.0, n_obs=0)
        assert result == 0.0, f"n_obs=0 must return 0.0, got {result}"

    def test_psr_positive_sharpe_above_benchmark_returns_above_half(self):
        from src.engine.ml_engineer_agent import MLEngineerAgent
        result = MLEngineerAgent._compute_psr(
            observed_sr=1.0, n_obs=252, sr_benchmark=0.0,
        )
        assert result > 0.5, f"Positive SR vs zero benchmark must give PSR > 0.5, got {result}"

    def test_psr_result_within_probability_bounds(self):
        from src.engine.ml_engineer_agent import MLEngineerAgent
        for observed_sr in [-5.0, 0.0, 2.0, 10.0]:
            result = MLEngineerAgent._compute_psr(observed_sr=observed_sr, n_obs=100)
            assert 0.0 <= result <= 1.0, (
                f"PSR must be a probability in [0,1], got {result} for observed_sr={observed_sr}"
            )

    def test_psr_not_computed_when_oos_sharpe_missing(self, tmp_path):
        """Post-review: PSR now requires `oos_sharpe` + `n_test` in meta JSON.
        Without them, the agent emits a WARN and skips the PSR metric."""
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(
            json.dumps(_good_meta({})),
            encoding='utf-8',
        )
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('meta', '1h', meta_path)
        # PSR not in metrics when oos_sharpe is missing
        assert 'psr' not in decision.metrics, (
            "PSR must not appear in metrics when oos_sharpe is missing"
        )
        assert any('psr not evaluated' in w.lower() for w in decision.warnings), (
            "Expected a WARN about PSR not being evaluated"
        )


# ── Group 12: Persistence edge cases ─────────────────────────────────────────

class TestDecisionPersistence:
    """Edges 16 & 17: read-only dir survives gracefully; list capped at 500."""

    def test_persist_to_read_only_directory_does_not_crash(self, tmp_path):
        """Writing to a read-only dir must log an error but NOT raise."""
        import stat
        ro_dir = tmp_path / 'readonly'
        ro_dir.mkdir()
        decisions_file = ro_dir / 'decisions.json'
        # Make directory read-only on Windows (remove write permission)
        try:
            ro_dir.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            agent = _agent.__func__  # can't use the fixture helper; construct directly
        except Exception:
            pytest.skip("Cannot set read-only permissions in this environment")

        from src.engine.ml_engineer_agent import MLEngineerAgent
        # Route decisions to a path inside the read-only dir
        try:
            agent_ro = MLEngineerAgent(decisions_path=decisions_file)
            # This should NOT raise — it must silently log the error
            agent_ro.validate_training_request(
                model_type='base', timeframe='1h',
                config={
                    'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'max_bars': 12, 'use_t1_purging': True,
                },
            )
        except PermissionError:
            # Windows: __init__ tries .parent.mkdir() which may also fail on read-only
            # If it raises in __init__ that is acceptable; the post-persist path must not raise.
            pass
        except Exception as exc:
            pytest.fail(f"Agent raised unexpected exception on read-only path: {exc}")
        finally:
            ro_dir.chmod(stat.S_IRWXU)  # restore so tmp_path can be cleaned up

    def test_persist_decision_survives_write_failure_via_mock(self, tmp_path):
        """When write_json raises, _persist_decision must catch and not propagate."""
        from src.engine.ml_engineer_agent import MLEngineerAgent
        agent = MLEngineerAgent(decisions_path=tmp_path / 'decisions.json')

        with patch('src.engine.ml_engineer_agent.write_json', side_effect=OSError("disk full")):
            # Must NOT raise
            try:
                agent.validate_training_request(
                    model_type='base', timeframe='1h',
                    config={
                        'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                        'max_bars': 12, 'use_t1_purging': True,
                    },
                )
            except Exception as exc:
                pytest.fail(f"Agent propagated write failure: {exc}")

    def test_decisions_capped_at_500(self, tmp_path):
        """After 501 decisions, only the most recent 500 are kept."""
        from src.engine.ml_engineer_agent import MLEngineerAgent, MLEngineerDecision
        from src.utils.safe_json import write_json

        decisions_path = tmp_path / 'decisions.json'
        # Pre-populate with 500 decisions
        existing = {'decisions': [
            {'timestamp': f'2024-01-01T00:0{i % 10}:00Z',
             'phase': 'pre_flight', 'model_type': 'base', 'timeframe': '1h',
             'decision': 'APPROVE', 'reasons': [], 'warnings': [], 'metrics': {}, 'config': {}}
            for i in range(500)
        ]}
        write_json(str(decisions_path), existing)

        agent = MLEngineerAgent(decisions_path=decisions_path)
        # Trigger one more persist
        agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                'max_bars': 12, 'use_t1_purging': True,
            },
        )

        from src.utils.safe_json import read_json
        result = read_json(str(decisions_path))
        assert len(result['decisions']) == 500, (
            f"Decisions list must be capped at 500, got {len(result['decisions'])}"
        )

    def test_decisions_list_grows_from_zero(self, tmp_path):
        """Starting from an empty file, each call adds exactly one decision."""
        agent = _agent(tmp_path)
        decisions_path = tmp_path / 'decisions.json'

        for i in range(3):
            agent.validate_training_request(
                model_type='base', timeframe='1h',
                config={
                    'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'max_bars': 12, 'use_t1_purging': True,
                },
            )

        from src.utils.safe_json import read_json
        result = read_json(str(decisions_path))
        assert len(result['decisions']) == 3


# ── Group 13: Post-training — non-meta model general KPI path ─────────────────

class TestPostTrainingGeneralKPI:
    """Non-meta models go through _check_general_kpi — must not crash."""

    def test_base_model_accept_when_above_coin_flip(self, tmp_path):
        meta = {
            'accuracy': 55.0,
            'walk_forward_mean_acc': 55.0,
            'walk_forward_std_acc': 2.0,
            'walk_forward_folds': 3,
            'n_features': 10,
            'n_train': 800,
            # Post-review: PSR gate now requires oos_sharpe + n_test, else WARN.
            'oos_sharpe': 1.2,
            'n_test': 200,
        }
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('base', '1h', meta_path)
        assert decision.decision == 'ACCEPT', (
            f"base model at 55% WF acc should ACCEPT, got {decision.decision}, "
            f"warnings={decision.warnings}"
        )

    def test_base_model_warn_when_below_coin_flip(self, tmp_path):
        meta = {
            'accuracy': 48.0,
            'walk_forward_mean_acc': 48.0,
            'walk_forward_std_acc': 0.0,
            'walk_forward_folds': 3,
            'n_features': 10,
            'n_train': 800,
        }
        meta_path = tmp_path / 'meta.json'
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        agent = _agent(tmp_path)
        decision = agent.evaluate_trained_model('trend', '4h', meta_path)
        # 48% < 50% → should at least warn
        assert decision.decision in ('FLAG_FOR_REVIEW', 'ACCEPT'), (
            f"Below coin-flip model should FLAG_FOR_REVIEW or trigger warning"
        )
        # The warning must be present
        assert any('coin' in w.lower() or '48' in w for w in decision.warnings)


# ── Group 14: Pre-flight — calibration and class_weight warnings ──────────────

class TestPreFlightPipelineConfig:
    """Non-standard calibration/class_weight should emit WARN, not BLOCK."""

    def test_wrong_calibration_method_warns_not_blocks(self, tmp_path):
        agent = _agent(tmp_path)
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5, 'sl_multiplier': 1.5, 'max_bars': 12,
                'use_t1_purging': True,
                'calibration_method': 'platt',   # not 'isotonic'
            },
        )
        calib_warnings = [w for w in decision.warnings if 'calibration' in w.lower()]
        assert calib_warnings, "Non-standard calibration_method must emit a warning"
        assert not any('BLOCK' in r for r in decision.reasons if 'calibration' in r.lower())

    def test_wrong_class_weight_warns_not_blocks(self, tmp_path):
        agent = _agent(tmp_path)
        # NOTE: The guard in _check_pipeline_config is `cw is not None and cw != 'balanced'`.
        # Passing None is treated as "not specified" (no warning). Use a concrete wrong
        # string value to test the warning path.
        decision = agent.validate_training_request(
            model_type='base', timeframe='1h',
            config={
                'pt_multiplier': 2.5, 'sl_multiplier': 1.5, 'max_bars': 12,
                'use_t1_purging': True,
                'class_weight': 'uniform',   # not 'balanced' and not None
            },
        )
        cw_warnings = [w for w in decision.warnings if 'class_weight' in w.lower()]
        assert cw_warnings, "Non-standard class_weight string must emit a warning"
        assert not any('BLOCK' in r for r in decision.reasons if 'class_weight' in r.lower())


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
