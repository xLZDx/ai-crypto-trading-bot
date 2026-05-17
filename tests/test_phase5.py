"""
Phase 5 tests — Level 5 Institutional Safeguards.

Coverage:
  - slippage_model: linear_slippage_bps, book_walk_slippage, real_price
  - beta_neutrality: BetaNeutralityFilter end-to-end
  - order_manager.circuit_breaker_check (DD, latency, data feed)
  - risk_agent: attach_beta_filter / check_beta_neutrality

Run:
    python tests/test_phase5.py
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


# ─── Slippage model ─────────────────────────────────────────────────────────

def test_slippage_model():
    print("\n[Slippage Model]")
    try:
        from src.analysis.slippage_model import (
            linear_slippage_bps, book_walk_slippage, real_price,
            total_execution_cost_bps, BookWalkResult,
        )
    except Exception as exc:
        check("import slippage_model", False, str(exc))
        return
    check("import slippage_model", True)

    # Linear formula sanity
    s = linear_slippage_bps(size=1.0, book_volume=100.0, lambda_impact=0.5)
    check("linear_slippage_bps(1, 100, 0.5) == 50",
          abs(s - 50.0) < 1e-9, f"got {s}")
    s2 = linear_slippage_bps(size=2.0, book_volume=100.0, lambda_impact=0.5)
    check("slippage scales linearly with size",
          abs(s2 - 2 * s) < 1e-9)

    # Empty book → max slippage
    huge = linear_slippage_bps(size=1.0, book_volume=0.0)
    check("zero book volume → 100% slip", huge == 1e4)

    # Book walk
    asks = [(100.0, 0.5), (101.0, 1.0), (102.0, 2.0)]
    bids = [(99.0, 1.0), (98.0, 1.0)]
    res = book_walk_slippage(size=1.0, side="buy", bids=bids, asks=asks, mid_price=99.5)
    check("book_walk fills 1.0", abs(res.fill_size - 1.0) < 1e-9)
    # 0.5 @100 + 0.5 @101 = avg 100.5
    check("book_walk avg price 100.5",
          abs(res.avg_price - 100.5) < 1e-6, f"got {res.avg_price}")
    check("book_walk slippage_bps positive", res.slippage_bps > 0)

    # real_price formula
    rp_buy = real_price(p_mid=100.0, side="buy", size=1.0, book_volume=100.0,
                        fee_bps=10, lambda_impact=0.5)
    # cost = 50 (slip) + 10 (fee) = 60 bps = 0.006 → 100 * 1.006 = 100.6
    check("real_price buy = mid * (1 + cost)",
          abs(rp_buy - 100.6) < 1e-6, f"got {rp_buy}")
    rp_sell = real_price(p_mid=100.0, side="sell", size=1.0, book_volume=100.0,
                         fee_bps=10, lambda_impact=0.5)
    check("real_price sell = mid * (1 - cost)",
          abs(rp_sell - 99.4) < 1e-6, f"got {rp_sell}")


# ─── Beta-neutrality filter ─────────────────────────────────────────────────

def test_beta_neutrality():
    print("\n[Beta Neutrality]")
    try:
        from src.analysis.beta_neutrality import BetaNeutralityFilter, BetaSnapshot
    except Exception as exc:
        check("import beta_neutrality", False, str(exc))
        return
    check("import beta_neutrality", True)

    import numpy as np, pandas as pd
    rng = np.random.default_rng(123)
    n = 500
    # Construct: ETH ≈ 1.2 × BTC + noise; SOL ≈ 1.5 × BTC; LINK ≈ 0.8 × BTC
    btc = rng.normal(0, 0.02, n)
    eth = 1.2 * btc + rng.normal(0, 0.005, n)
    sol = 1.5 * btc + rng.normal(0, 0.01, n)
    lnk = 0.8 * btc + rng.normal(0, 0.005, n)
    history = pd.DataFrame({"BTC/USDT": btc, "ETH/USDT": eth,
                            "SOL/USDT": sol, "LINK/USDT": lnk})

    bn = BetaNeutralityFilter(history, factor="BTC/USDT", max_beta_exposure=1.0)
    snap0 = bn.snapshot()
    check("snapshot returns BetaSnapshot", isinstance(snap0, BetaSnapshot))
    check("β(BTC/BTC) == 1", abs(bn._betas["BTC/USDT"] - 1.0) < 1e-9)
    check("β(ETH) ≈ 1.2 ± 0.1",
          abs(bn._betas["ETH/USDT"] - 1.2) < 0.1,
          f"got {bn._betas['ETH/USDT']:.3f}")
    check("β(SOL) ≈ 1.5 ± 0.1",
          abs(bn._betas["SOL/USDT"] - 1.5) < 0.1,
          f"got {bn._betas['SOL/USDT']:.3f}")

    # Empty book — no breach for low-β asset (LINK β≈0.8 < cap=1.0)
    check("empty book: no breach for fresh low-β long",
          not bn.would_breach("LINK/USDT", "long", notional=1_000))
    # ETH (β≈1.2) on its own SHOULD breach the cap=1.0
    check("empty book: high-β single position correctly breaches cap",
          bn.would_breach("ETH/USDT", "long", notional=1_000))

    # Add big SOL long, then test if more long would breach
    bn.update_position("SOL/USDT", side="long", notional=10_000)
    snap = bn.snapshot()
    check("aggregate β > 0 after long SOL", snap.aggregate_beta > 0)
    # Adding more longs in the same factor direction should breach
    check("adding ETH long after big SOL long would breach",
          bn.would_breach("ETH/USDT", "long", notional=20_000))
    # An offsetting short should NOT breach
    check("offsetting ETH short does not breach",
          not bn.would_breach("ETH/USDT", "short", notional=5_000))


# ─── order_manager circuit breaker ──────────────────────────────────────────

def test_circuit_breaker():
    print("\n[Order Manager -- circuit breakers]")
    src = (PROJECT_ROOT / "src" / "engine" / "order_manager.py").read_text(encoding="utf-8")
    check("circuit_breaker_check() defined", "def circuit_breaker_check" in src)
    for trigger in ("max_daily_drawdown", "api_latency", "data_feed_inconsistency"):
        check(f"trigger {trigger} present", trigger in src)


# ─── risk_agent beta gate ───────────────────────────────────────────────────

def test_risk_agent_beta_gate():
    print("\n[risk_agent -- ?-neutrality gate]")
    src = (PROJECT_ROOT / "src" / "engine" / "agents" / "risk_agent.py").read_text(encoding="utf-8")
    check("attach_beta_filter() defined", "def attach_beta_filter" in src)
    check("check_beta_neutrality() defined", "def check_beta_neutrality" in src)
    check("imports BetaNeutralityFilter",
          "from src.analysis.beta_neutrality import BetaNeutralityFilter" in src)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 5 -- Level 5 Institutional Safeguards Tests")
    print("=" * 60)
    test_slippage_model()
    test_beta_neutrality()
    test_circuit_breaker()
    test_risk_agent_beta_gate()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
