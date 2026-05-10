"""
Order Flow Transformer (OFT) — Phase 2, Level 2 (Alpha Engine).

Architecture per updated_architecture_plan_en.md §6:

    Event Embedding -> Order Book Encoder -> Temporal Transformer -> Cross-Attention

Multi-task output:
    - Return distribution  (μ, log σ²)
    - Probability of movement  (binary "TP-vs-SL" head)
    - Liquidity risk  (continuous, sigmoid-bounded ∈ [0,1])

The model is built on PyTorch — no darts/neuralforecast dependency. It can be
trained with the joint OFT+RL objective from Phase 3:

    L = -E[PnL] + λ1·CVaR + λ2·ImpactCost + λ3·InventoryRisk

For now (Phase 2) we expose just the supervised head. Calibration is applied
post-hoc by `src.training.oft_trainer.IsotonicCalibrator`.

Inputs are two streams:
    events     : (B, T_e, F_e)   tick-level events (trades, cancels, deltas)
    orderbook  : (B, T_o, F_o)   regularly-sampled L2 snapshots

T_e and T_o may differ; cross-attention aligns them. Both can be padded.

This file deliberately keeps the model small (`d_model=128`, `n_layers=2`) for
Phase 2 sanity tests. Joint training in Phase 3 will scale up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import NamedTuple

# PyTorch is required for this module — but we make the import lazy so the
# module can be imported in environments without torch (e.g. for unit tests
# that only check the symbols exist).
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    _TORCH_OK = False


@dataclass
class OFTConfig:
    """Hyperparameters. Sensible defaults for sanity / Phase 2 size."""
    event_features:     int = 16   # OFI, taker imbalance, trade size, etc.
    orderbook_features: int = 8    # imbalance, microprice, depth, ...
    d_model:            int = 128
    n_heads:            int =   4
    n_layers:           int =   2
    ff_dim:             int = 256
    dropout:            float = 0.1
    max_event_len:      int = 256
    max_orderbook_len:  int = 256
    n_regimes:          int =   3   # for regime-conditional embedding
    use_regime_cond:    bool = True


class OFTOutput(NamedTuple):
    """Multi-task forward output."""
    mu:             "torch.Tensor"   # (B,) expected log-return
    log_var:        "torch.Tensor"   # (B,) log of σ² for return
    p_move:         "torch.Tensor"   # (B,) sigmoid prob of TP-vs-SL = 1
    liquidity_risk: "torch.Tensor"   # (B,) sigmoid liquidity-risk score


def _sinusoidal_pe(max_len: int, d_model: int, device=None):
    """Standard absolute positional encoding."""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                    -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)  # (1, T, D)


if _TORCH_OK:

    class _EventEmbedding(nn.Module):
        """Project per-event feature vectors into d_model + add positional code."""

        def __init__(self, in_dim: int, d_model: int, max_len: int, dropout: float):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            self.register_buffer("pe", _sinusoidal_pe(max_len, d_model), persistent=False)
            self.drop = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):  # (B, T, F)
            T = x.size(1)
            h = self.proj(x) + self.pe[:, :T, :]
            return self.norm(self.drop(h))

    class _OrderBookEncoder(nn.Module):
        """Encode regularly-sampled L2 snapshots with a self-attention stack."""

        def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int,
                     ff_dim: int, max_len: int, dropout: float):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            self.register_buffer("pe", _sinusoidal_pe(max_len, d_model), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        def forward(self, x, key_padding_mask=None):
            T = x.size(1)
            h = self.proj(x) + self.pe[:, :T, :]
            return self.encoder(h, src_key_padding_mask=key_padding_mask)

    class _TemporalTransformer(nn.Module):
        """Self-attention over the event stream."""

        def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        def forward(self, x, key_padding_mask=None):
            return self.encoder(x, src_key_padding_mask=key_padding_mask)

    class OrderFlowTransformer(nn.Module):
        """Hybrid Transformer: events × order book → multi-task output."""

        def __init__(self, cfg: OFTConfig | None = None):
            super().__init__()
            self.cfg = cfg or OFTConfig()
            c = self.cfg

            self.event_emb = _EventEmbedding(c.event_features, c.d_model,
                                             c.max_event_len, c.dropout)
            self.ob_encoder = _OrderBookEncoder(c.orderbook_features, c.d_model,
                                                c.n_heads, c.n_layers, c.ff_dim,
                                                c.max_orderbook_len, c.dropout)
            self.temporal = _TemporalTransformer(c.d_model, c.n_heads, c.n_layers,
                                                 c.ff_dim, c.dropout)
            # Cross-attention: event tokens query order-book tokens
            self.cross = nn.MultiheadAttention(embed_dim=c.d_model, num_heads=c.n_heads,
                                               dropout=c.dropout, batch_first=True)
            self.norm_xattn = nn.LayerNorm(c.d_model)

            self.regime_emb = (nn.Embedding(c.n_regimes, c.d_model)
                               if c.use_regime_cond else None)

            # Multi-task heads
            self.head_mu      = nn.Linear(c.d_model, 1)
            self.head_log_var = nn.Linear(c.d_model, 1)
            self.head_p_move  = nn.Linear(c.d_model, 1)
            self.head_liq     = nn.Linear(c.d_model, 1)

        def forward(
            self,
            events,                 # (B, T_e, F_e)
            orderbook,              # (B, T_o, F_o)
            event_mask=None,        # (B, T_e) — True where padded
            orderbook_mask=None,    # (B, T_o)
            regime=None,            # (B,) long
        ) -> OFTOutput:
            ev = self.event_emb(events)
            ob = self.ob_encoder(orderbook, key_padding_mask=orderbook_mask)
            ev = self.temporal(ev, key_padding_mask=event_mask)

            x_attn, _ = self.cross(query=ev, key=ob, value=ob,
                                   key_padding_mask=orderbook_mask, need_weights=False)
            ev = self.norm_xattn(ev + x_attn)

            # Pool: masked-mean over event tokens
            if event_mask is not None:
                m = (~event_mask).float().unsqueeze(-1)  # (B,T,1)
                pooled = (ev * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = ev.mean(dim=1)

            if self.regime_emb is not None and regime is not None:
                pooled = pooled + self.regime_emb(regime)

            mu      = self.head_mu(pooled).squeeze(-1)
            log_var = self.head_log_var(pooled).squeeze(-1)
            p_move  = torch.sigmoid(self.head_p_move(pooled).squeeze(-1))
            liq     = torch.sigmoid(self.head_liq(pooled).squeeze(-1))
            return OFTOutput(mu=mu, log_var=log_var, p_move=p_move, liquidity_risk=liq)

        # ── Loss helpers ────────────────────────────────────────────────────

        @staticmethod
        def gaussian_nll(out: OFTOutput, y_return: "torch.Tensor") -> "torch.Tensor":
            """Negative log-likelihood under N(μ, σ²) — supervises (μ, log_var)."""
            var = torch.exp(out.log_var).clamp(min=1e-6)
            return 0.5 * ((y_return - out.mu) ** 2 / var + out.log_var).mean()

        @staticmethod
        def bce_p_move(out: OFTOutput, y_binary: "torch.Tensor") -> "torch.Tensor":
            return F.binary_cross_entropy(out.p_move.clamp(1e-6, 1 - 1e-6), y_binary.float())

        @staticmethod
        def total_loss(out: OFTOutput, y_return, y_binary, y_liq=None,
                       w_nll=1.0, w_bce=1.0, w_liq=0.5) -> "torch.Tensor":
            l = w_nll * OrderFlowTransformer.gaussian_nll(out, y_return) \
              + w_bce * OrderFlowTransformer.bce_p_move(out, y_binary)
            if y_liq is not None:
                l = l + w_liq * F.binary_cross_entropy(
                    out.liquidity_risk.clamp(1e-6, 1 - 1e-6), y_liq.float()
                )
            return l

else:  # torch unavailable — provide a stub that raises on instantiation

    class OrderFlowTransformer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError(
                "OrderFlowTransformer needs torch. Install it via "
                "`pip install torch torchvision torchaudio --index-url "
                "https://download.pytorch.org/whl/cu118`."
            )


__all__ = ["OrderFlowTransformer", "OFTConfig", "OFTOutput"]
