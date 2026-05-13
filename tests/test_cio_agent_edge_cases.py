"""
CIO Agent — Edge-Case Behavioral Tests.

Tests beyond what exists in test_phase0_fix.py. Every test here:
  1. Calls the function under test directly.
  2. Asserts on observable behavior (return value, side-effects, raised exception).
  3. Mocks optuna with unittest.mock where the real library is not needed
     (avoids pulling in SQLite / real study creation in every test).

Coverage targets:
  - Constructor: lazy optuna import (no crash without package)
  - _ensure_optuna: missing optuna → clear RuntimeError
  - objective: smoke-test mode (no task_submitter) → deterministic stub
  - objective: pt/sl ratio < 1.3 → TrialPruned raised
  - _compute_psr: benchmark=0, mean>0 → probability > 0.5
  - apply_best: operator_approved=False → {'ok': False, 'error': ...}
  - apply_best: operator_approved=True → {'ok': True, ...}
  - _persist_proposal: list capped at 50 entries
  - Singleton behaviour: get_cio_agent() returns same instance
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trial(number: int = 0, suggest_categorical=None,
                suggest_int=None, suggest_float=None) -> MagicMock:
    """Build a minimal Optuna Trial mock."""
    trial = MagicMock()
    trial.number = number
    trial.suggest_categorical = suggest_categorical or (
        lambda name, choices: choices[0]
    )
    trial.suggest_int = suggest_int or (
        lambda name, lo, hi, step=1: lo
    )
    trial.suggest_float = suggest_float or (
        lambda name, lo, hi, step=0.1: lo
    )
    trial.set_user_attr = MagicMock()
    trial.should_prune = MagicMock(return_value=False)
    trial.report = MagicMock()
    return trial


def _make_optuna_module() -> types.ModuleType:
    """Build a fake 'optuna' module with just enough API surface."""
    opt = types.ModuleType('optuna')

    class _TrialPruned(Exception):
        pass

    class _Exceptions:
        TrialPruned = _TrialPruned

    opt.exceptions = _Exceptions()

    class _TPESampler:
        def __init__(self, **kwargs): pass

    class _MedianPruner:
        def __init__(self, **kwargs): pass

    class _Samplers:
        TPESampler = _TPESampler

    class _Pruners:
        MedianPruner = _MedianPruner

    opt.samplers = _Samplers()
    opt.pruners = _Pruners()

    return opt


# ── Group 1: Constructor — lazy import ────────────────────────────────────────

class TestConstructor:
    """Edge 1: CIOAgent constructs without optuna installed."""

    def test_constructor_does_not_import_optuna(self):
        """_optuna must remain None until _ensure_optuna() is called."""
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(study_name='edge_test_lazy')
        assert agent._optuna is None, (
            "_optuna must be None immediately after construction (lazy import)"
        )

    def test_constructor_stores_study_name(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(study_name='my_study')
        assert agent.study_name == 'my_study'

    def test_constructor_stores_max_dd_threshold(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(max_dd_threshold=0.25)
        assert agent.max_dd_threshold == 0.25

    def test_constructor_ml_engineer_gate_defaults_true(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent()
        assert agent.ml_engineer_gate is True

    def test_constructor_with_optuna_missing_does_not_raise(self):
        """Even if optuna is not importable, the constructor must succeed."""
        original = sys.modules.get('optuna')
        sys.modules['optuna'] = None  # simulate ImportError on import
        try:
            from importlib import reload
            import src.engine.cio_agent as cio_mod
            reload(cio_mod)
            agent = cio_mod.CIOAgent(study_name='no_optuna_test')
            assert agent is not None
        finally:
            if original is None:
                sys.modules.pop('optuna', None)
            else:
                sys.modules['optuna'] = original


# ── Group 2: _ensure_optuna — missing package ─────────────────────────────────

class TestEnsureOptuna:
    """Edge 2: _ensure_optuna when optuna is missing → raises clear RuntimeError."""

    def test_ensure_optuna_raises_runtime_error_when_missing(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent()
        # Patch the import so it raises ImportError
        with patch.dict(sys.modules, {'optuna': None}):
            with pytest.raises(RuntimeError, match="optuna not installed"):
                agent._ensure_optuna()

    def test_ensure_optuna_returns_module_when_available(self):
        from src.engine.cio_agent import CIOAgent
        fake_optuna = _make_optuna_module()
        agent = CIOAgent()
        with patch.dict(sys.modules, {'optuna': fake_optuna}):
            result = agent._ensure_optuna()
        assert result is fake_optuna

    def test_ensure_optuna_caches_on_second_call(self):
        from src.engine.cio_agent import CIOAgent
        fake_optuna = _make_optuna_module()
        agent = CIOAgent()
        with patch.dict(sys.modules, {'optuna': fake_optuna}):
            r1 = agent._ensure_optuna()
            r2 = agent._ensure_optuna()
        assert r1 is r2, "_ensure_optuna must return the same cached module"


# ── Group 3: objective — smoke-test mode ─────────────────────────────────────

class TestObjectiveSmokeTestMode:
    """Edge 3: task_submitter=None → deterministic stub value, no crash."""

    def _make_agent_with_fake_optuna(self, fake_optuna, **kwargs):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(task_submitter=None, ml_engineer_gate=False, **kwargs)
        agent._optuna = fake_optuna
        return agent

    def test_objective_smoke_mode_returns_deterministic_value(self):
        fake_optuna = _make_optuna_module()
        agent = self._make_agent_with_fake_optuna(fake_optuna)

        trial = _make_trial(number=3)
        # Ensure pt/sl ratio ≥ 1.3 so AFML prune is not triggered
        trial.suggest_float = lambda name, lo, hi, step=0.1: {
            'pt_multiplier': 2.5,
            'sl_multiplier': 1.5,
            'confidence_threshold': 0.55,
        }[name]

        result = agent.objective(trial)
        expected = float(3 % 7) * 0.1
        assert result == expected, (
            f"Smoke-test mode must return float(trial.number % 7) * 0.1, "
            f"expected {expected}, got {result}"
        )

    def test_objective_smoke_mode_is_float(self):
        fake_optuna = _make_optuna_module()
        agent = self._make_agent_with_fake_optuna(fake_optuna)
        trial = _make_trial(number=0)
        trial.suggest_float = lambda name, lo, hi, step=0.1: {
            'pt_multiplier': 2.5,
            'sl_multiplier': 1.5,
            'confidence_threshold': 0.55,
        }[name]
        result = agent.objective(trial)
        assert isinstance(result, float), f"objective must return float, got {type(result)}"

    def test_objective_smoke_mode_trial_number_7_wraps(self):
        """trial.number=7 → 7 % 7 = 0 → returns 0.0"""
        fake_optuna = _make_optuna_module()
        agent = self._make_agent_with_fake_optuna(fake_optuna)
        trial = _make_trial(number=7)
        trial.suggest_float = lambda name, lo, hi, step=0.1: {
            'pt_multiplier': 2.5,
            'sl_multiplier': 1.5,
            'confidence_threshold': 0.55,
        }[name]
        result = agent.objective(trial)
        assert result == 0.0


# ── Group 4: objective — TrialPruned on pt/sl ratio < 1.3 ────────────────────

class TestObjectiveTrialPruned:
    """Edge 8: pt/sl ratio < 1.3 → optuna.TrialPruned raised."""

    def test_low_pt_sl_ratio_prunes_trial(self):
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(task_submitter=None, ml_engineer_gate=False)
        agent._optuna = fake_optuna

        # Force pt/sl = 2.0/2.0 = 1.0 < 1.3 → must prune
        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.0, 'sl_multiplier': 2.0,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=1)
        trial.suggest_float = suggest_float

        with pytest.raises(fake_optuna.exceptions.TrialPruned):
            agent.objective(trial)

    def test_ratio_exactly_1_3_not_pruned(self):
        """pt/sl = 2.6/2.0 = 1.3 (exactly) must NOT prune."""
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(task_submitter=None, ml_engineer_gate=False)
        agent._optuna = fake_optuna

        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.6, 'sl_multiplier': 2.0,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=2)
        trial.suggest_float = suggest_float
        # ratio = 1.3 exactly → NOT pruned → smoke-test stub returns float
        result = agent.objective(trial)
        assert isinstance(result, float)

    def test_ratio_just_below_1_3_prunes(self):
        """pt=2.5, sl=2.0 → ratio=1.25 < 1.3 → prune."""
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(task_submitter=None, ml_engineer_gate=False)
        agent._optuna = fake_optuna

        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.5, 'sl_multiplier': 2.0,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=3)
        trial.suggest_float = suggest_float

        with pytest.raises(fake_optuna.exceptions.TrialPruned):
            agent.objective(trial)


# ── Group 5: _compute_psr ─────────────────────────────────────────────────────

class TestComputePSR:
    """Edge 4: sr_benchmark=0.0, mean>0 → PSR > 0.5."""

    def test_psr_mean_above_zero_benchmark_returns_above_half(self):
        from src.engine.ml_engineer_agent import MLEngineerAgent
        # Reuse the same static method from the ML Engineer (both agents share the math)
        result = MLEngineerAgent._compute_psr(mean_sr=1.0, std_sr=0.2, n_folds=6,
                                               sr_benchmark=0.0)
        assert result > 0.5, f"mean=1.0 > benchmark=0.0 must give PSR > 0.5, got {result}"

    def test_cio_agent_does_not_expose_compute_psr_but_ml_engineer_does(self):
        """CIO delegate check: CIOAgent relies on MLEngineerAgent for PSR gate."""
        from src.engine.cio_agent import CIOAgent
        assert not hasattr(CIOAgent, '_compute_psr'), (
            "CIOAgent should not duplicate _compute_psr; delegate to MLEngineerAgent"
        )


# ── Group 6: apply_best — operator approval gate ─────────────────────────────

class TestApplyBest:
    """Edges 5 & 6: operator_approved=False → ok:False; True → ok:True."""

    def test_apply_best_without_approval_returns_error(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent()
        result = agent.apply_best(operator_approved=False)
        assert result.get('ok') is False, (
            f"apply_best without approval must return ok=False, got {result}"
        )
        assert 'error' in result
        assert result['error'], "error message must be non-empty"

    def test_apply_best_without_approval_does_not_modify_rules(self, tmp_path):
        """Must not touch training_rules.json when not approved.

        apply_best() with operator_approved=False must return early immediately,
        so the real TRAINING_RULES_PATH is never written. We verify this by
        placing a sentinel file at the real path (via monkeypatch) and asserting
        it stays untouched.
        """
        rules_path = tmp_path / 'training_rules.json'
        rules_path.write_text(json.dumps({'original': True}), encoding='utf-8')

        from src.engine.cio_agent import CIOAgent
        import src.engine.cio_agent as cio_mod

        agent = CIOAgent()
        # Redirect the module-level constant so that IF apply_best were to write,
        # it would write to our tmp file (giving us a chance to detect it).
        original_path = cio_mod.TRAINING_RULES_PATH
        cio_mod.TRAINING_RULES_PATH = rules_path
        try:
            agent.apply_best(operator_approved=False)
        finally:
            cio_mod.TRAINING_RULES_PATH = original_path

        data = json.loads(rules_path.read_text(encoding='utf-8'))
        assert data.get('original') is True, (
            "training_rules.json must not be modified when operator_approved=False"
        )

    def test_apply_best_with_approval_returns_ok_true(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent()
        result = agent.apply_best(operator_approved=True)
        assert result.get('ok') is True, (
            f"apply_best with approval must return ok=True, got {result}"
        )

    def test_apply_best_default_approval_is_false(self):
        """Ensure the default parameter is False (no accidental auto-apply)."""
        import inspect
        from src.engine.cio_agent import CIOAgent
        sig = inspect.signature(CIOAgent.apply_best)
        default = sig.parameters['operator_approved'].default
        assert default is False, (
            f"operator_approved default must be False, got {default!r}"
        )


# ── Group 7: _persist_proposal — list capped at 50 ───────────────────────────

class TestPersistProposal:
    """Edge 7: _persist_proposal keeps only the last 50 proposals."""

    def test_proposal_list_capped_at_50(self, tmp_path):
        from src.engine.cio_agent import CIOAgent
        from src.utils.safe_json import write_json, read_json

        proposals_path = tmp_path / 'cio_proposals.json'
        # Pre-populate with 50 proposals
        existing = {'proposals': [
            {'study_name': f's{i}', 'n_trials': i, 'best_value': float(i)}
            for i in range(50)
        ]}
        write_json(str(proposals_path), existing)

        agent = CIOAgent()
        with patch('src.engine.cio_agent.CIO_PROPOSALS_PATH', proposals_path):
            agent._persist_proposal({'study_name': 'new_study', 'n_trials': 99, 'best_value': 9.9})

        data = read_json(str(proposals_path))
        assert len(data['proposals']) == 50, (
            f"Proposals must be capped at 50, got {len(data['proposals'])}"
        )
        # The LATEST proposal must be the one we just added
        assert data['proposals'][-1]['study_name'] == 'new_study'

    def test_proposal_list_grows_from_empty(self, tmp_path):
        from src.engine.cio_agent import CIOAgent
        from src.utils.safe_json import read_json

        proposals_path = tmp_path / 'cio_proposals.json'
        agent = CIOAgent()

        with patch('src.engine.cio_agent.CIO_PROPOSALS_PATH', proposals_path):
            for i in range(3):
                agent._persist_proposal({'study_name': f's{i}', 'best_value': float(i)})

        data = read_json(str(proposals_path))
        assert len(data['proposals']) == 3

    def test_persist_proposal_survives_write_failure(self, tmp_path):
        """If write_json fails, _persist_proposal must catch and not propagate.

        _persist_proposal imports write_json inside the method body
        (`from src.utils.safe_json import write_json`) so we must patch
        write_json at its definition site in src.utils.safe_json.
        """
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent()

        with patch('src.utils.safe_json.write_json', side_effect=OSError("no space")):
            try:
                agent._persist_proposal({'study_name': 'fail_test', 'best_value': 0.0})
            except Exception as exc:
                pytest.fail(f"_persist_proposal propagated write failure: {exc}")


# ── Group 8: Singleton ────────────────────────────────────────────────────────

class TestSingleton:
    """get_cio_agent() must return the same instance on repeated calls."""

    def test_get_cio_agent_returns_same_instance(self):
        import src.engine.cio_agent as cio_mod
        # Reset singleton to avoid cross-test contamination
        cio_mod._cio_singleton = None
        a1 = cio_mod.get_cio_agent(study_name='singleton_test')
        a2 = cio_mod.get_cio_agent()
        assert a1 is a2, "get_cio_agent() must return the same cached instance"
        cio_mod._cio_singleton = None  # clean up after test

    def test_get_cio_agent_accepts_kwargs_on_first_call(self):
        import src.engine.cio_agent as cio_mod
        cio_mod._cio_singleton = None
        agent = cio_mod.get_cio_agent(study_name='kwarg_test', max_dd_threshold=0.20)
        assert agent.study_name == 'kwarg_test'
        assert agent.max_dd_threshold == 0.20
        cio_mod._cio_singleton = None


# ── Group 9: objective — ML Engineer gate integration ────────────────────────

class TestObjectiveMLEngineerGate:
    """Verify the ML Engineer BLOCK → TrialPruned path in objective."""

    def test_ml_engineer_block_causes_trial_pruned(self):
        """ML Engineer BLOCK → TrialPruned.

        `get_ml_engineer` is imported INLINE inside objective() via:
            from src.engine.ml_engineer_agent import get_ml_engineer
        so we must patch it at the definition site, not at the cio_agent namespace.
        """
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent

        agent = CIOAgent(task_submitter=None, ml_engineer_gate=True)
        agent._optuna = fake_optuna

        # Build a mock pre-flight decision that is BLOCK
        mock_decision = MagicMock()
        mock_decision.decision = 'BLOCK'
        mock_decision.reasons = ['BLOCK: test block']

        mock_ml_agent = MagicMock()
        mock_ml_agent.validate_training_request.return_value = mock_decision

        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=5)
        trial.suggest_float = suggest_float

        # Patch at the ml_engineer_agent module — where the function is defined —
        # because objective() uses a local `from` import.
        with patch('src.engine.ml_engineer_agent.get_ml_engineer', return_value=mock_ml_agent):
            with pytest.raises(fake_optuna.exceptions.TrialPruned):
                agent.objective(trial)

    def test_ml_engineer_warn_does_not_prune(self):
        """ML Engineer WARN does not prune the trial — only BLOCK does."""
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent

        agent = CIOAgent(task_submitter=None, ml_engineer_gate=True)
        agent._optuna = fake_optuna

        mock_decision = MagicMock()
        mock_decision.decision = 'WARN'
        mock_decision.reasons = []

        mock_ml_agent = MagicMock()
        mock_ml_agent.validate_training_request.return_value = mock_decision

        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=6)
        trial.suggest_float = suggest_float

        with patch('src.engine.ml_engineer_agent.get_ml_engineer', return_value=mock_ml_agent):
            result = agent.objective(trial)
        # WARN does not prune → smoke-test stub still returns a float
        assert isinstance(result, float)

    def test_ml_engineer_exception_does_not_prune(self):
        """If ML Engineer itself raises, the trial proceeds (failsafe)."""
        fake_optuna = _make_optuna_module()
        from src.engine.cio_agent import CIOAgent

        agent = CIOAgent(task_submitter=None, ml_engineer_gate=True)
        agent._optuna = fake_optuna

        mock_ml_agent = MagicMock()
        mock_ml_agent.validate_training_request.side_effect = RuntimeError("agent unavailable")

        def suggest_float(name, lo, hi, step=0.1):
            return {'pt_multiplier': 2.5, 'sl_multiplier': 1.5,
                    'confidence_threshold': 0.55}[name]

        trial = _make_trial(number=7)
        trial.suggest_float = suggest_float

        with patch('src.engine.ml_engineer_agent.get_ml_engineer', return_value=mock_ml_agent):
            result = agent.objective(trial)
        assert isinstance(result, float), (
            "ML Engineer failure must be swallowed, trial should not crash"
        )


# ── Group 10: _poll_task_completion — no poller ───────────────────────────────

class TestPollTaskCompletion:
    """When task_status_poller is None, _poll_task_completion returns a default dict."""

    def test_poll_returns_default_dict_when_no_poller(self):
        from src.engine.cio_agent import CIOAgent
        agent = CIOAgent(task_status_poller=None)
        trial = _make_trial()
        result = agent._poll_task_completion('fake_task_id', trial=trial, timeout_s=1)
        assert result is not None
        assert 'sortino' in result
        assert result['sortino'] == 0.0


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
