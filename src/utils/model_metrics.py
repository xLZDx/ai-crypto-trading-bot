"""
model_metrics — shared post-training metric computation.

Pre-PR-44 only the meta-labeler wrote `auc_roc` and `win_precision` to its
meta JSON. The dashboard's Model Training table has columns for both, so
every other model showed `—` for AUC and Win Prec% — looked like missing
data even though all our HistGBT classifiers are perfectly capable of
producing those numbers.

This helper centralises the computation so every trainer writes the same
fields. Output is a dict that can be merged into each trainer's meta
dict before write_json.

AUC is the right discrimination signal even for accuracy-tuned
classifiers — it answers "if I picked one positive and one negative at
random, would the model rank the positive higher?" — independent of
whatever threshold the operator settles on.

Win precision is "of the trades where the model's calibrated probability
≥ THRESHOLD, what fraction actually won". Closer to operator intuition
("if I trust the model when it says ≥0.6, what win rate do I get") than
raw accuracy.
"""
from __future__ import annotations

from typing import Sequence


# Default high-confidence threshold for win-precision. Matches the
# meta-labeler's confidence_threshold default.
DEFAULT_WIN_PREC_THRESHOLD = 0.6


def compute_classifier_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    *,
    threshold: float = DEFAULT_WIN_PREC_THRESHOLD,
) -> dict:
    """Return {auc_roc, win_precision, win_rate_pct, n_high_conf} for a
    binary classifier. y_true is 0/1 ground-truth labels; y_proba is the
    predicted probability of class 1 (e.g. from calibrated HistGBT's
    predict_proba(X)[:, 1]).

    Failure modes that return None instead of raising:
      * y_true has only one class — AUC undefined
      * no samples crossed the threshold — win_precision undefined

    All values pre-multiplied by 100 to match the meta-labeler's
    convention (so the dashboard column shows "46.3%" not "0.463").
    auc_roc is the exception — the dashboard formats it directly with
    `.toFixed(3)` and expects 0.0–1.0 range.
    """
    import numpy as np

    out: dict = {
        'auc_roc':       None,
        'win_precision': None,
        'win_rate_pct':  None,
        'n_high_conf':   0,
        'threshold':     threshold,
    }
    try:
        y_true_arr  = np.asarray(y_true).astype(int)
        y_proba_arr = np.asarray(y_proba).astype(float)
    except Exception:
        return out
    if y_true_arr.size == 0 or y_proba_arr.size == 0:
        return out

    # AUC — only defined when both classes present in y_true.
    if len(set(y_true_arr.tolist())) >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            out['auc_roc'] = float(roc_auc_score(y_true_arr, y_proba_arr))
        except Exception:
            pass

    # Win precision at threshold.
    high_conf = y_proba_arr >= threshold
    out['n_high_conf'] = int(high_conf.sum())
    if high_conf.any():
        wins = (y_true_arr[high_conf] == 1).sum()
        out['win_precision'] = float(wins / high_conf.sum() * 100.0)

    # Overall win rate (label=1 frequency in test set) — sanity check
    # the operator can compare against win_precision. If win_precision
    # ≈ win_rate_pct, the threshold isn't filtering useful signal.
    out['win_rate_pct'] = float((y_true_arr == 1).mean() * 100.0)

    return out


def merge_metrics_into_meta(
    meta: dict,
    y_true: Sequence[int],
    y_proba: Sequence[float],
    *,
    threshold: float = DEFAULT_WIN_PREC_THRESHOLD,
) -> dict:
    """Convenience wrapper — compute metrics + merge into meta dict in
    place. Skips a key if its value is None so we don't overwrite a
    real value with a missing one. Returns the same meta for chaining.
    """
    metrics = compute_classifier_metrics(y_true, y_proba, threshold=threshold)
    for k, v in metrics.items():
        if v is not None:
            meta[k] = v
    return meta
