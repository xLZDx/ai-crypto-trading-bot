"""
Phase 4 tests — Level 4 Portfolio Optimization.

Coverage:
  - cvar_optimizer.confidence_weights / risk_parity_weights / CVaROptimizer
  - dynamic_threshold.find_best_threshold / rolling_threshold
  - kelly_criterion.kelly_weight_prior
  - risk_manager.cvar_position_weights

Run:
    python tests/test_phase4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def check(name, ok, detail=""):
    if ok is None:
        results["skip"] += 1
        print(f"  {SKIP} {name} (skipped)")
    elif ok:
        results["pass"] += 1
        print(f"  {PASS} {name}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}{': ' + detail if detail else ''}")


# ─── CVaR optimizer ─────────────────────────────────────────────────────────

def test_cvar_optimizer():
    print("\n[CVaR Optimizer]")
    try:
        from src.analysis.cvar_optimizer import (
            CVaROptimizer, CVaRResult, confidence_weights, risk_parity_weights,
        )
    except Exception as exc:
        check("import cvar_optimizer", False, str(exc))
        return
    check("import cvar_optimizer", True)

    import numpy as np, pandas as pd
    rng = np.random.default_rng(2026)

    # confidence_weights ranges
    cw = confidence_weights(np.array([0.0, 0.5, 1.0]))
    check("confidence_weights(0)==-1, (0.5)==0, (1)==1",
          abs(cw[0] + 1) < 1e-9 and abs(cw[1]) < 1e-9 and abs(cw[2] - 1) < 1e-9)

    # risk_parity_weights
    probs = np.array([0.7, 0.55, 0.45])
    vols  = np.array([0.02, 0.04, 0.01])
    rp = risk_parity_weights(probs, vols)
    check("risk_parity_weights normalised (sum |w| ≈ 1)",
          abs(np.sum(np.abs(rp)) - 1.0) < 1e-9)

    # CVaROptimizer on synthetic returns
    n_scen, n_assets = 500, 4
    scen = rng.normal(0.001, 0.02, size=(n_scen, n_assets))
    # Inject one asset with much heavier tail
    scen[:, 3] = rng.standard_t(df=3, size=n_scen) * 0.05
    opt = CVaROptimizer(alpha=0.05, lam=1.0, leverage_cap=1.0, box_max=0.6)
    res = opt.fit(scen)
    check("CVaR result returned", isinstance(res, CVaRResult))
    check("weights sum |w| <= leverage_cap + tol",
          np.sum(np.abs(res.weights)) <= 1.0 + 1e-3,
          f"sum={np.sum(np.abs(res.weights)):.4f}")
    check("each |w_i| <= box_max + tol",
          np.all(np.abs(res.weights) <= 0.6 + 1e-3))
    check("status is optimal or close",
          "optimal" in res.status.lower() or "solved" in res.status.lower(),
          f"status={res.status}")
    # Heavy-tail asset (idx 3) should be downweighted vs equal-prior assets
    check("CVaR shrinks heavy-tail asset (|w_3| <= max(|w_others|))",
          abs(res.weights[3]) <= max(abs(res.weights[:3])) + 1e-3)


# ─── Dynamic threshold ──────────────────────────────────────────────────────

def test_dynamic_threshold():
    print("\n[Dynamic Threshold]")
    try:
        from src.analysis.dynamic_threshold import (
            find_best_threshold, rolling_threshold, ThresholdSearchResult,
        )
    except Exception as exc:
        check("import dynamic_threshold", False, str(exc))
        return
    check("import dynamic_threshold", True)

    import numpy as np, pandas as pd
    rng = np.random.default_rng(7)
    n = 500
    # Construct: when prob > 0.6, returns are systematically positive
    probs = rng.uniform(0, 1, size=n)
    returns = np.where(probs > 0.6, rng.normal(0.005, 0.01, n),
                                     rng.normal(-0.001, 0.01, n))

    res = find_best_threshold(probs, returns, grid_low=0.5, grid_high=0.8, grid_n=15)
    check("returns ThresholdSearchResult", isinstance(res, ThresholdSearchResult))
    check("best_threshold ∈ [0.5, 0.8]",
          0.5 <= res.best_threshold <= 0.8)
    check("best_threshold biased above 0.55 (signal-aware)",
          res.best_threshold >= 0.55,
          f"thr={res.best_threshold}")
    check("grid has 15 entries", len(res.grid) == 15)

    # rolling
    p_s = pd.Series(probs)
    r_s = pd.Series(returns)
    rolling = rolling_threshold(p_s, r_s, window=200, refit_every=50)
    check("rolling_threshold same length as input", len(rolling) == n)
    check("rolling first window stays at 0.5 default",
          float(rolling.iloc[0]) == 0.5)


# ─── Kelly weight prior ─────────────────────────────────────────────────────

def test_kelly_prior():
    print("\n[Kelly Weight Prior]")
    try:
        from src.analysis.kelly_criterion import kelly_weight_prior, MAX_KELLY_FRACTION
    except Exception as exc:
        check("import kelly_weight_prior", False, str(exc))
        return
    check("import kelly_weight_prior", True)

    import numpy as np
    p = np.array([0.55, 0.7, 0.45, 0.4])
    w = kelly_weight_prior(p)
    check("output length matches input", len(w) == 4)
    check("each weight in [0, MAX_KELLY_FRACTION]",
          np.all((w >= 0) & (w <= MAX_KELLY_FRACTION + 1e-9)))
    check("higher p → larger weight", w[1] >= w[0] >= w[3])


# ─── risk_manager CVaR helper ───────────────────────────────────────────────

def test_risk_manager_cvar():
    print("\n[Risk Manager — CVaR helper]")
    src = (PROJECT_ROOT / "src" / "analysis" / "risk_manager.py").read_text(encoding="utf-8")
    check("cvar_position_weights() defined", "def cvar_position_weights" in src)
    check("imports CVaROptimizer", "from src.analysis.cvar_optimizer import CVaROptimizer" in src)
    check("imports kelly_weight_prior",
          "from src.analysis.kelly_criterion import kelly_weight_prior" in src)


# ─── requirements.txt — cvxpy ───────────────────────────────────────────────

def test_requirements():
    print("\n[requirements.txt — Phase 4]")
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    check("cvxpy in requirements.txt", "cvxpy" in req)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 4 — Level 4 Portfolio Optimization Tests")
    print("=" * 60)
    test_requirements()
    test_cvar_optimizer()
    test_dynamic_threshold()
    test_kelly_prior()
    test_risk_manager_cvar()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
