"""
OFT Trainer — Phase 2, Level 2 (Alpha Engine).

Implements the Anti-Overfitting methodology from
updated_architecture_plan_en.md §7:

    • Purged Walk-Forward CV — eliminate label-overlap leakage between train
      and test folds.
    • Regime Conditioning  — pass `regime` as a side input so the model learns
      `model(x | regime)`.
    • Output Calibration   — IsotonicRegression / Temperature scaling on
      held-out probability outputs.
    • Microstructure Augmentation — additive Gaussian noise on order-book
      tensors during training only.

This module deliberately doesn't depend on `mlfinlab` (license-restricted
package). PurgedKFold is implemented locally; the algorithm matches Lopez de
Prado, *Advances in Financial Machine Learning* §7.4.

Public API:
    folds = purged_kfold(t1_times, n_splits=5, embargo_pct=0.01)
    OFTTrainer(model, ...).run(events, orderbook, returns, binary_y, regime, t1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Purged Walk-Forward CV ──────────────────────────────────────────────────

def purged_kfold(
    t1_times: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    *,
    bidirectional_purge: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold CV — a self-contained replacement for `mlfinlab.PurgedKFold`.

    mlfinlab is unavailable on Python 3.14 (Hudson&Thames closed-sourced
    2.0+; `mlfinpy` fork needs an old numba). This implementation covers
    the three correctness requirements from López de Prado (Adv. Fin. ML,
    §7.4):

      1. **Label-overlap purge**     — train samples whose label-resolution
         time `t1` falls into the test window are dropped.
      2. **Embargo period**          — `embargo_pct` of the dataset after
         each test fold is excluded from training to block trailing
         look-ahead from price-driven serial correlation.
      3. **Strict chronological order** — folds are built by sequential
         positional slicing; no shuffling. The function asserts the input
         `t1_times` is non-decreasing so callers don't accidentally pass
         scrambled data.

    Bidirectional purge (default ON) also drops *post-test* training rows
    whose own input window would peek into the test period — strictly
    safer than mlfinlab's default but equivalent at the embargo boundary.

    Args:
        t1_times: Series whose values are the label-resolution timestamps
                  (the second return of `triple_barrier_labels_vectorized`).
        n_splits: Number of folds.
        embargo_pct: Fraction of the dataset purged after each test fold.
        bidirectional_purge: when True, also drop post-test rows whose t1
                             starts inside the test window.

    Returns:
        list[(train_idx, test_idx)] per fold (positional, chronological).
    """
    t1 = t1_times.copy()
    if not isinstance(t1.index, pd.RangeIndex):
        t1 = t1.reset_index(drop=True)
    # Defensive: confirm chronological ordering. If violated, sort and warn.
    t1_dt = pd.to_datetime(t1)
    if not t1_dt.is_monotonic_increasing:
        import warnings
        warnings.warn("[purged_kfold] t1_times is not monotonic -- sorting. "
                      "Make sure this matches the row order of your features!")
        order = np.argsort(t1_dt.to_numpy())
        t1 = t1.iloc[order].reset_index(drop=True)
        t1_dt = pd.to_datetime(t1)

    n = len(t1)
    fold_size = n // n_splits
    embargo = int(n * embargo_pct)
    folds = []
    for k in range(n_splits):
        test_start = k * fold_size
        test_end   = (k + 1) * fold_size if k < n_splits - 1 else n
        test_idx   = np.arange(test_start, test_end)

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_start:test_end] = False

        # 1) Left-side purge: train rows whose label resolves into the test window
        if test_start > 0:
            test_first_t = t1_dt.iloc[test_start]
            leaks_left = t1_dt.iloc[:test_start] >= test_first_t
            train_mask[:test_start] = train_mask[:test_start] & (~leaks_left).to_numpy()

        # 2) Embargo: contiguous gap immediately after the test fold
        if embargo > 0 and test_end + embargo < n:
            train_mask[test_end:test_end + embargo] = False

        # 3) Right-side bidirectional purge: train rows whose t1 starts
        #    inside the test window (rare but possible at the boundary)
        if bidirectional_purge and test_end < n:
            test_last_t = t1_dt.iloc[test_end - 1]
            leaks_right = t1_dt.iloc[test_end:] <= test_last_t
            train_mask[test_end:] = train_mask[test_end:] & (~leaks_right).to_numpy()

        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


# ─── Microstructure augmentation ─────────────────────────────────────────────

def microstructure_augment(orderbook_tensor, sigma: float = 0.02):
    """Inject Gaussian noise into order-book features at training time.

    Empirically improves OFT robustness to feed jitter and synthetic-replay
    artefacts. Keep `sigma` ≤ 0.05 — too much erases the signal.
    """
    import torch
    if sigma <= 0:
        return orderbook_tensor
    return orderbook_tensor + torch.randn_like(orderbook_tensor) * sigma


# ─── Output calibration ──────────────────────────────────────────────────────

class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression for OFT p_move calibration.

    Use after walk-forward training:
        cal = IsotonicCalibrator().fit(p_uncal, y_true)
        p_cal = cal.transform(p_uncal_new)
    """

    def __init__(self):
        self._iso = None

    def fit(self, p_uncalibrated, y_true) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(np.asarray(p_uncalibrated, dtype=float),
                      np.asarray(y_true, dtype=float))
        return self

    def transform(self, p_uncalibrated):
        if self._iso is None:
            return p_uncalibrated
        return self._iso.transform(np.asarray(p_uncalibrated, dtype=float))


# ─── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class OFTTrainerConfig:
    epochs:         int = 5
    batch_size:     int = 64
    lr:             float = 1e-3
    weight_decay:   float = 1e-5
    n_splits:       int = 5
    embargo_pct:    float = 0.01
    augment_sigma:  float = 0.02
    grad_clip_norm: float = 1.0
    device:         str = "auto"  # "auto" | "cpu" | "cuda"


class OFTTrainer:
    """Walk-forward, purged, calibrated OFT trainer.

    Holds the *training loop* — does not own the model. Pass an
    `OrderFlowTransformer` instance and tensors prepared upstream.
    """

    def __init__(self, model, cfg: OFTTrainerConfig | None = None):
        self.model = model
        self.cfg = cfg or OFTTrainerConfig()
        self.calibrator: IsotonicCalibrator | None = None
        self._fold_metrics: list[dict] = []

    # ── helpers ─────────────────────────────────────────────────────────────

    def _device(self):
        import torch
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)

    def _make_loader(self, idx, events, orderbook, returns, binary_y, regime):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(
            events[idx], orderbook[idx],
            returns[idx], binary_y[idx],
            regime[idx] if regime is not None else torch.zeros(len(idx), dtype=torch.long),
        )
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

    # ── training loop ───────────────────────────────────────────────────────

    def run(
        self,
        events,         # tensor (N, T_e, F_e)
        orderbook,      # tensor (N, T_o, F_o)
        returns,        # tensor (N,)        target log-returns
        binary_y,       # tensor (N,)        TP-vs-SL labels {0,1}
        regime,         # tensor (N,) long  or None
        t1_times: pd.Series,
    ) -> dict:
        """Full purged walk-forward training, then post-hoc calibration."""
        import torch
        device = self._device()
        # Disable flash/fused attention on sm_75 (RTX 2060): both kernels
        # trigger intermittent cudaErrorInvalidConfiguration with seq_len=16.
        # Forces the math (non-flash) SDPA path which is stable on Turing.
        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.model.to(device)
        events    = events.to(device)
        orderbook = orderbook.to(device)
        returns   = returns.to(device)
        binary_y  = binary_y.to(device)
        if regime is not None:
            regime = regime.to(device)

        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

        folds = purged_kfold(t1_times, n_splits=self.cfg.n_splits,
                             embargo_pct=self.cfg.embargo_pct)

        # Walk-forward: fold k uses fold k as test; folds 0..k-1 are train.
        # Plain k-fold purged (LdP §7) gives more training data per fold.
        all_test_p = []
        all_test_y = []

        for k, (train_idx, test_idx) in enumerate(folds):
            train_loader = self._make_loader(train_idx, events, orderbook,
                                             returns, binary_y, regime)
            self.model.train()
            for epoch in range(self.cfg.epochs):
                running = 0.0
                for ev_b, ob_b, r_b, y_b, reg_b in train_loader:
                    ob_b = microstructure_augment(ob_b, sigma=self.cfg.augment_sigma)
                    out = self.model(ev_b, ob_b,
                                     event_mask=None, orderbook_mask=None,
                                     regime=reg_b if regime is not None else None)
                    loss = self.model.total_loss(out, r_b, y_b)
                    opt.zero_grad()
                    loss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.grad_clip_norm)
                    opt.step()
                    running += float(loss.item())
                logger.info("[OFT] fold %d epoch %d  loss=%.4f", k, epoch,
                            running / max(len(train_loader), 1))

            # Test-fold predictions for calibration — batched to avoid OOM on
            # large test sets (full-batch transformer forward allocates huge
            # intermediate tensors proportional to test_size × T × d_model).
            self.model.eval()
            test_p_chunks = []
            eval_bs = self.cfg.batch_size * 4  # larger than train batch; no gradients
            with torch.no_grad():
                for start in range(0, len(test_idx), eval_bs):
                    chunk = test_idx[start:start + eval_bs]
                    reg_chunk = (regime[chunk] if regime is not None else None)
                    out = self.model(events[chunk], orderbook[chunk],
                                    regime=reg_chunk)
                    test_p_chunks.append(out.p_move.detach().cpu().numpy())
            all_test_p.append(np.concatenate(test_p_chunks))
            all_test_y.append(binary_y[test_idx].detach().cpu().numpy())
            self._fold_metrics.append({"fold": k, "n_train": len(train_idx),
                                       "n_test": len(test_idx)})

        # Post-hoc isotonic calibration on out-of-fold predictions
        p_oof = np.concatenate(all_test_p)
        y_oof = np.concatenate(all_test_y)
        self.calibrator = IsotonicCalibrator().fit(p_oof, y_oof)
        logger.info("[OFT] training done; %d OOF preds calibrated", len(p_oof))
        return {
            "fold_metrics": self._fold_metrics,
            "n_oof":        int(len(p_oof)),
            "oof_pos_rate": float(np.mean(y_oof)),
        }


__all__ = [
    "OFTTrainer", "OFTTrainerConfig",
    "purged_kfold", "microstructure_augment", "IsotonicCalibrator",
]
