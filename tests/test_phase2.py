"""
Phase 2 tests — Level 2 Alpha Engine.

Coverage:
  - event_time_labeler.regime_normalized_barriers / label_event_time
  - models.order_flow_transformer.OrderFlowTransformer  (forward + losses)
  - training.oft_trainer.purged_kfold / IsotonicCalibrator
  - regime_classifier upgraded to BayesianGaussianMixture
  - inference_engine OFT path symbols present

Run:
    python tests/test_phase2.py
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


# ─── Event-time labeler ──────────────────────────────────────────────────────

def test_event_time_labeler():
    print("\n[Event-Time Labeler]")
    try:
        from src.analysis.event_time_labeler import (
            label_event_time, regime_normalized_barriers,
            filter_for_binary_classification, EventTimeLabels,
        )
    except Exception as exc:
        check("import event_time_labeler", False, str(exc))
        return
    check("import event_time_labeler", True)

    import numpy as np, pandas as pd
    rng = np.random.default_rng(11)
    n = 600
    close = 30000 + rng.normal(0, 50, n).cumsum()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "open":  close + rng.normal(0, 5, n),
        "high":  close + rng.normal(20, 5, n),
        "low":   close + rng.normal(-20, 5, n),
        "close": close,
        "volume": rng.uniform(1, 5, n),
    })
    tp, sl, vn = regime_normalized_barriers(df, k_tp=2.0, k_sl=2.0)
    check("barriers shape matches input", tp.shape == (n,) and sl.shape == (n,))
    check("vol_norm finite, > 0",
          np.all(np.isfinite(vn)) and np.all(vn >= 0))

    out = label_event_time(df, k_tp=2.0, k_sl=2.0, max_horizon_bars=120)
    check("label_event_time returns EventTimeLabels", isinstance(out, EventTimeLabels))
    check("labels series length matches input", len(out.labels) == n)
    check("stats has all expected keys",
          all(k in out.stats for k in ("n", "long_pct", "short_pct", "timeout_pct", "binary_n")))
    check("binary_y is timeout-free",
          (out.binary_y.isin([0, 1])).all())

    # Filtering helper
    X = pd.DataFrame({"feat": np.arange(n)}, index=df.index)
    Xf, yf = filter_for_binary_classification(X, out)
    check("filter shapes match",
          len(Xf) == len(yf) == out.stats["binary_n"])


# ─── Order Flow Transformer ──────────────────────────────────────────────────

def test_order_flow_transformer():
    print("\n[Order Flow Transformer]")
    try:
        import torch
    except ImportError:
        check("torch available", None)
        return

    try:
        from src.models.order_flow_transformer import (
            OrderFlowTransformer, OFTConfig, OFTOutput,
        )
    except Exception as exc:
        check("import OrderFlowTransformer", False, str(exc))
        return
    check("import OrderFlowTransformer", True)

    cfg = OFTConfig(
        event_features=8, orderbook_features=4,
        d_model=32, n_heads=2, n_layers=1, ff_dim=64,
        max_event_len=20, max_orderbook_len=20,
    )
    model = OrderFlowTransformer(cfg)
    check("model instantiates with reduced cfg", model is not None)

    B, Te, To = 4, 20, 20
    ev = torch.randn(B, Te, cfg.event_features)
    ob = torch.randn(B, To, cfg.orderbook_features)
    regime = torch.zeros(B, dtype=torch.long)

    out = model(ev, ob, regime=regime)
    check("forward returns OFTOutput", isinstance(out, OFTOutput))
    check("mu shape == (B,)", out.mu.shape == (B,))
    check("p_move ∈ [0,1]",
          float(out.p_move.min()) >= 0 and float(out.p_move.max()) <= 1)
    check("liquidity_risk ∈ [0,1]",
          float(out.liquidity_risk.min()) >= 0 and float(out.liquidity_risk.max()) <= 1)

    # Losses
    y_ret  = torch.randn(B)
    y_bin  = torch.randint(0, 2, (B,))
    nll = OrderFlowTransformer.gaussian_nll(out, y_ret)
    bce = OrderFlowTransformer.bce_p_move(out, y_bin)
    tot = OrderFlowTransformer.total_loss(out, y_ret, y_bin)
    check("gaussian_nll is finite scalar",
          torch.isfinite(nll).item() and nll.dim() == 0)
    check("bce_p_move is finite scalar",
          torch.isfinite(bce).item() and bce.dim() == 0)
    check("total_loss is finite scalar",
          torch.isfinite(tot).item() and tot.dim() == 0)

    # Backprop sanity
    tot.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
    check("backward populates gradients", has_grad)


# ─── OFT Trainer / PurgedKFold / Calibrator ──────────────────────────────────

def test_oft_trainer():
    print("\n[OFT Trainer]")
    try:
        from src.training.oft_trainer import (
            purged_kfold, IsotonicCalibrator, microstructure_augment,
            OFTTrainer, OFTTrainerConfig,
        )
    except Exception as exc:
        check("import oft_trainer", False, str(exc))
        return
    check("import oft_trainer", True)

    import numpy as np, pandas as pd
    n = 200
    base = pd.date_range("2025-01-01", periods=n, freq="1h")
    t1 = pd.Series(base + pd.Timedelta(minutes=30))
    folds = purged_kfold(t1, n_splits=5, embargo_pct=0.02)
    check("purged_kfold returns 5 folds", len(folds) == 5)

    train_idx, test_idx = folds[2]
    check("no overlap between train and test",
          len(set(train_idx) & set(test_idx)) == 0)

    # Forced overlap leak — verify purge kicks in
    t1_leak = pd.Series(base + pd.Timedelta(hours=200))  # extreme leak
    folds_leak = purged_kfold(t1_leak, n_splits=5)
    train_leak, test_leak = folds_leak[2]
    check("with strong t1 leak, train shrinks below default",
          len(train_leak) < n - len(test_leak))

    # Isotonic calibration sanity
    cal = IsotonicCalibrator()
    p_uncal = np.linspace(0, 1, 100)
    y = (p_uncal > 0.5).astype(int)
    cal.fit(p_uncal, y)
    p_cal = cal.transform([0.1, 0.5, 0.9])
    check("calibrator transform produces 3 values",
          len(p_cal) == 3)
    check("calibrator output ∈ [0,1]",
          0.0 <= float(min(p_cal)) and float(max(p_cal)) <= 1.0)

    # Microstructure augmentation
    try:
        import torch
        x = torch.zeros(4, 5, 3)
        x_aug = microstructure_augment(x, sigma=0.1)
        check("microstructure_augment changes the tensor",
              not torch.equal(x_aug, x))
    except ImportError:
        check("microstructure_augment with torch", None)


# ─── Regime classifier (BayesianGaussianMixture) ────────────────────────────

def test_regime_classifier():
    print("\n[Regime Classifier — BayesianGaussianMixture]")
    src_path = PROJECT_ROOT / "src" / "analysis" / "regime_classifier.py"
    src = src_path.read_text(encoding="utf-8")
    check("uses BayesianGaussianMixture",
          "from sklearn.mixture import BayesianGaussianMixture" in src)
    check("weight_concentration_prior=0.01 (per arch plan)",
          "weight_concentration_prior=0.01" in src)
    check("uses dirichlet_process",
          "weight_concentration_prior_type=\"dirichlet_process\"" in src
          or "weight_concentration_prior_type='dirichlet_process'" in src)
    check("partial_fit() method present", "def partial_fit" in src)
    check("model_type metadata == BayesianGaussianMixture",
          "BayesianGaussianMixture" in src and "model_type" in src)


# ─── Inference engine OFT path ───────────────────────────────────────────────

def test_inference_engine_oft():
    print("\n[Inference Engine — OFT path]")
    src_path = PROJECT_ROOT / "src" / "engine" / "inference_engine.py"
    src = src_path.read_text(encoding="utf-8")
    check("_load_oft_model() defined", "_load_oft_model" in src)
    check("_oft_predict() defined", "_oft_predict" in src)
    check("OrderFlowTransformer imported in OFT loader",
          "OrderFlowTransformer" in src)
    check("predictions surface 'oft' key", '"oft"' in src or "'oft'" in src)
    check("oft prediction has mu, sigma, p_move",
          all(k in src for k in ("\"mu\"", "\"p_move\"", "\"sigma\"")))


# ─── Runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Phase 2 — Level 2 Alpha Engine Tests")
    print("=" * 60)
    test_event_time_labeler()
    test_order_flow_transformer()
    test_oft_trainer()
    test_regime_classifier()
    test_inference_engine_oft()
    total = sum(results.values())
    print("\n" + "=" * 60)
    print(f"PASS: {results['pass']}  FAIL: {results['fail']}  SKIP: {results['skip']}  TOTAL: {total}")
    print("=" * 60)
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
