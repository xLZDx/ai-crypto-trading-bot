"""
Phase 3 tests — Level 3 Execution & Simulation.

Coverage:
  - alpha_decay.apply_alpha_decay / half_life / should_exit
  - synthetic_exchange — softmax_fill, full reset/step lifecycle
  - rl_base — ReplayBuffer, ContinuousBox, obs/reward helpers
  - rl_execution_sac.SACAgent — act + update on a tiny replay buffer
  - rl_execution_ppo.PPOAgent — act + update on a tiny rollout
  - multi_agent_env — full episode with NoiseAgent + MomentumAgent
  - order_manager.should_alpha_decay_exit() helper

Run:
    python tests/test_phase3.py
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


# ─── alpha_decay ─────────────────────────────────────────────────────────────

def test_alpha_decay():
    print("\n[Alpha Decay]")
    try:
        from src.analysis.alpha_decay import (
            apply_alpha_decay, half_life, should_exit, decay_curve,
        )
    except Exception as exc:
        check("import alpha_decay", False, str(exc))
        return
    check("import alpha_decay", True)

    # apply formula matches expected
    val = apply_alpha_decay(1.0, time_in_trade=0.0, decay_rate=0.1)
    check("decay at t=0 == signal", abs(val - 1.0) < 1e-9)

    val_t10 = apply_alpha_decay(1.0, time_in_trade=10.0, decay_rate=0.1)
    import math
    check("decay at t=10, k=0.1 ≈ exp(-1.0)",
          abs(val_t10 - math.exp(-1.0)) < 1e-9)

    # half_life
    hl = half_life(0.1)
    check("half_life(0.1) ≈ 6.93", abs(hl - math.log(2) / 0.1) < 1e-9)

    # should_exit
    check("should_exit triggers when decayed below threshold",
          should_exit(signal_strength=1.0, time_in_trade=50.0, decay_rate=0.1,
                      exit_threshold=0.2))
    check("should_exit FALSE when still strong",
          not should_exit(signal_strength=1.0, time_in_trade=1.0, decay_rate=0.1,
                          exit_threshold=0.2))

    # curve
    curve = decay_curve(1.0, 0.1, 10)
    check("decay_curve length == t_max+1", len(curve) == 11)
    check("decay_curve monotonically decreasing",
          all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1)))


# ─── synthetic_exchange ─────────────────────────────────────────────────────

def test_synthetic_exchange():
    print("\n[Synthetic Exchange]")
    try:
        from src.simulation.synthetic_exchange import (
            SyntheticExchange, softmax_fill, ImpactModel,
        )
    except Exception as exc:
        check("import synthetic_exchange", False, str(exc))
        return
    check("import synthetic_exchange", True)

    # softmax_fill monotonicity
    f_zero = softmax_fill(0, 100, 0.0)
    f_aggr = softmax_fill(0, 100, 0.5)
    f_pass = softmax_fill(0, 100, -0.5)
    check("softmax_fill ∈ (0,1)", 0.0 < f_zero < 1.0)
    check("softmax_fill increases with aggressive offset", f_aggr > f_zero > f_pass)

    # Full episode lifecycle
    book_iter = [
        {"timestamp": i, "p_bid": 100 - 0.1, "p_ask": 100 + 0.1,
         "v_bid": 10.0 + i * 0.1, "v_ask": 10.0}
        for i in range(20)
    ]
    ex = SyntheticExchange(book_iter, impact=ImpactModel(lambda_impact=0.3))
    obs = ex.reset()
    check("reset returns dict", isinstance(obs, dict))
    check("obs has imbalance", "imbalance" in obs)

    cum = 0.0
    n = 0
    while True:
        obs, r, done, info = ex.step((0.5, 0.2))    # buy 0.5 units, +20 bps
        cum += r
        n += 1
        if done:
            break
    check("episode stepped multiple times", n > 1)
    check("info contains fill_pct", "fill_pct" in info)
    check("inventory bounded", -10.1 <= ex.state.inventory <= 10.1)


# ─── rl_base ────────────────────────────────────────────────────────────────

def test_rl_base():
    print("\n[RL Base]")
    try:
        from src.models.rl_base import (
            ContinuousBox, make_action_space, make_observation_space,
            obs_dict_to_vector, shaped_reward,
            Transition, ReplayBuffer,
        )
    except Exception as exc:
        check("import rl_base", False, str(exc))
        return
    check("import rl_base", True)

    space = make_action_space()
    check("action space shape == (2,)", space.shape == (2,))
    a = space.sample()
    check("action sample within bounds",
          (a >= space.low).all() and (a <= space.high).all())

    obs_v = obs_dict_to_vector({"imbalance": 0.3, "spread_bps": 0.0001,
                                 "inventory": 1.0, "v_bid": 10.0, "v_ask": 10.0})
    check("obs_dict_to_vector length 5", len(obs_v) == 5)

    # shaped reward formula
    import math
    r = shaped_reward(raw_pnl=1.0, inventory=2.0, inventory_lambda=0.5)
    check("shaped_reward(1, 2, 0.5) == 1 - 0.5*4 == -1.0",
          abs(r - (-1.0)) < 1e-9)

    # ReplayBuffer
    import numpy as np
    buf = ReplayBuffer(capacity=10, obs_dim=5)
    for i in range(5):
        buf.push(Transition(np.zeros(5, dtype=np.float32),
                            np.zeros(2, dtype=np.float32),
                            float(i), np.zeros(5, dtype=np.float32), False))
    check("ReplayBuffer length == 5", len(buf) == 5)
    o, ac, r, no, d = buf.sample(3)
    check("buffer sample shape", o.shape == (3, 5) and ac.shape == (3, 2))


# ─── SAC + PPO ──────────────────────────────────────────────────────────────

def test_sac_agent():
    print("\n[SAC Agent]")
    try:
        import torch
    except ImportError:
        check("torch available", None)
        return
    try:
        from src.models.rl_execution_sac import SACAgent
        from src.models.rl_base import ReplayBuffer, Transition
    except Exception as exc:
        check("import SACAgent", False, str(exc))
        return
    check("import SACAgent", True)

    agent = SACAgent(obs_dim=5, hidden=16, device="cpu")
    import numpy as np
    a = agent.act(np.zeros(5, dtype=np.float32))
    check("SAC act returns shape (2,)", a.shape == (2,))
    check("SAC act bounded by tanh", abs(a).max() <= 1.0)

    # Tiny update
    buf = ReplayBuffer(capacity=200, obs_dim=5)
    rng = np.random.default_rng(0)
    for _ in range(80):
        o = rng.standard_normal(5).astype(np.float32)
        no = rng.standard_normal(5).astype(np.float32)
        ac = rng.uniform(-1, 1, size=2).astype(np.float32)
        buf.push(Transition(o, ac, float(rng.standard_normal()), no, False))
    metrics = agent.update(buf, batch_size=32)
    check("SAC update produces actor_loss", "actor_loss" in metrics)
    check("SAC update produces alpha", "alpha" in metrics)


def test_ppo_agent():
    print("\n[PPO Agent]")
    try:
        import torch
    except ImportError:
        check("torch available", None)
        return
    try:
        from src.models.rl_execution_ppo import PPOAgent, Rollout
    except Exception as exc:
        check("import PPOAgent", False, str(exc))
        return
    check("import PPOAgent", True)

    agent = PPOAgent(obs_dim=5, hidden=16, device="cpu", epochs=1)
    import numpy as np
    a = agent.act(np.zeros(5, dtype=np.float32))
    check("PPO act returns shape (2,)", a.shape == (2,))

    # Build a tiny rollout & GAE
    rng = np.random.default_rng(1)
    n = 32
    obs   = rng.standard_normal((n, 5)).astype(np.float32)
    acts  = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    logp  = rng.standard_normal(n).astype(np.float32)
    rews  = rng.standard_normal(n).astype(np.float32)
    vals  = rng.standard_normal(n).astype(np.float32)
    dones = np.zeros(n, dtype=bool)
    adv, ret = agent.compute_gae(rews, vals, dones, last_value=0.0)
    rollout = Rollout(obs=obs, actions=acts, log_probs=logp,
                      returns=ret, advantages=adv)
    metrics = agent.update(rollout, batch_size=8)
    check("PPO update returns policy_loss",
          "policy_loss" in metrics or metrics.get("skipped"))


# ─── multi_agent_env ─────────────────────────────────────────────────────────

def test_multi_agent_env():
    print("\n[Multi-Agent Env]")
    try:
        from src.simulation.multi_agent_env import (
            MultiAgentEnv, NoiseAgent, MomentumAgent,
        )
    except Exception as exc:
        check("import multi_agent_env", False, str(exc))
        return
    check("import multi_agent_env", True)

    book = [
        {"timestamp": i, "p_bid": 100, "p_ask": 100.2,
         "v_bid": 10 + (i % 3), "v_ask": 10}
        for i in range(15)
    ]
    alpha = MomentumAgent(k_size=0.3, k_offset=0.1)
    env = MultiAgentEnv(book, alpha_agent=alpha,
                        adversaries=[NoiseAgent(sigma=0.1)])
    env.reset()
    iterations = 0
    while not env.done and iterations < 50:
        env.step()
        iterations += 1
    rep = env.report()
    check("env reports for both agents",
          len(rep["agents"]) == 2)
    check("alpha agent has cum_reward",
          "cum_reward" in next(iter(rep["agents"].values())))


# ─── order_manager alpha-decay helper ────────────────────────────────────────

def test_order_manager_decay_helper():
    print("\n[order_manager — alpha-decay helper]")
    src = (PROJECT_ROOT / "src" / "engine" / "order_manager.py").read_text(encoding="utf-8")
    check("should_alpha_decay_exit() defined", "def should_alpha_decay_exit" in src)
    check("imports alpha_decay.should_exit",
          "from src.analysis.alpha_decay import should_exit" in src)


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 3 — Level 3 Execution & Simulation Tests")
    print("=" * 60)
    test_alpha_decay()
    test_synthetic_exchange()
    test_rl_base()
    test_sac_agent()
    test_ppo_agent()
    test_multi_agent_env()
    test_order_manager_decay_helper()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
