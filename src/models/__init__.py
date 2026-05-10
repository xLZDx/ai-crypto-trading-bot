"""
src.models — neural-network architectures for the institutional upgrade.

Phase 2: Order Flow Transformer (OFT) — multi-task model that consumes
event-stream + L2 order book snapshots and emits (μ, σ², p_move, liquidity_risk).

Phase 3: RL execution agents — SAC (primary) + PPO (backup).
"""
from .order_flow_transformer import OrderFlowTransformer, OFTConfig, OFTOutput
from .rl_base import (
    BaseExecutionAgent, ContinuousBox, ReplayBuffer, Transition,
    obs_dict_to_vector, shaped_reward, make_action_space,
)

# Lazy-import the torch-only RL agents so importing src.models on a
# torch-less host (e.g. for unit tests) doesn't crash.
def __getattr__(name):
    if name == "SACAgent":
        from .rl_execution_sac import SACAgent
        return SACAgent
    if name == "PPOAgent":
        from .rl_execution_ppo import PPOAgent
        return PPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OrderFlowTransformer", "OFTConfig", "OFTOutput",
    "BaseExecutionAgent", "ContinuousBox", "ReplayBuffer", "Transition",
    "obs_dict_to_vector", "shaped_reward", "make_action_space",
    "SACAgent", "PPOAgent",
]
