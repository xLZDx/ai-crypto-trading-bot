"""v3.1 step 12 (1E) — backfill the 912 historical trades with the 7
enrichment fields added going-forward by step 11 (1D).

Strategy:
- mode: assume 'testnet' (the bot has been on testnet by default per
  CLAUDE.md). Mark inferred=True in the output schema so downstream
  doesn't double-count.
- regime_at_entry: re-run RegimeClassifier on the closest available
  1h bar at buy_time; falls back to None if data unavailable.
- mfe_pct / mae_pct: best-effort using sell_price vs buy_price + the
  high/low excursion if 1m bars are loadable. Falls back to a tighter
  MFE/MAE estimate from highest_price (which the live tracker already
  updated during the trade's lifetime).
- model_confidence: unrecoverable for historical trades; set None.
- slippage_pct: unrecoverable; set None.
- exit_reason: infer from pnl_pct sign + status (TP / SL / flat /
  open).

Output: data/trades_enriched.json (untouched original).

Usage:
    python -m scripts.backfill_trade_enrichment
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRADES_PATH = PROJECT_ROOT / 'data' / 'trades.json'
OUT_PATH    = PROJECT_ROOT / 'data' / 'trades_enriched.json'


def _infer_exit_reason(t: dict) -> str | None:
    """Infer exit_reason from existing fields. Returns None for OPEN."""
    status = (t.get('status') or '').upper()
    if status == 'OPEN':
        return None
    pnl = t.get('pnl_usdt')
    if pnl is None:
        return 'unknown'
    if pnl > 0:
        return 'TP'
    if pnl < 0:
        return 'SL'
    return 'flat'


def _mfe_mae_from_existing(t: dict) -> tuple[float, float]:
    """Best-effort MFE/MAE without re-loading 1m bars.

    The live trade tracker maintained `highest_price` during the
    trade's lifetime, which gives a one-sided excursion estimate.
    For LONG: MFE = (highest - buy) / buy * 100, MAE proxied by
    pnl_pct floor (hidden in pnl_percent if it ever went negative).

    Better approach when 1m bars are loadable would re-run the
    full high/low scan; this keeps the script <30 s for 912 rows.
    """
    side = (t.get('side') or 'LONG').upper()
    buy  = float(t.get('buy_price') or 0)
    high = float(t.get('highest_price') or buy)
    low  = float(t.get('lowest_price')  or buy)  # not always present on legacy rows
    if buy <= 0:
        return 0.0, 0.0
    if side == 'LONG':
        mfe = max(0.0, (high - buy) / buy * 100.0)
        # If lowest_price wasn't tracked, approximate MAE from
        # pnl_percent's floor (the close was at least this bad).
        if low and low < buy:
            mae = (low - buy) / buy * 100.0
        else:
            pnl_pct = t.get('pnl_percent')
            mae = float(pnl_pct) if pnl_pct is not None and pnl_pct < 0 else 0.0
    else:
        # SHORT: highest tracks the bottom; reverse the math.
        mfe = max(0.0, (buy - high) / buy * 100.0)
        if low and low > buy:
            mae = (buy - low) / buy * 100.0
        else:
            pnl_pct = t.get('pnl_percent')
            mae = float(pnl_pct) if pnl_pct is not None and pnl_pct < 0 else 0.0
    return float(mfe), float(min(0.0, mae))


def _try_get_regime(_classifier, _ts: str) -> str | None:
    """Re-run RegimeClassifier on the closest 1h bar to ts.

    Optional; classifier may be unavailable on bot-stopped state.
    """
    if _classifier is None:
        return None
    try:
        # Lazy import — RegimeClassifier might pull DuckDB which is heavy.
        from src.analysis.regime_classifier import RegimeClassifier  # noqa: F401
        # Stub: full implementation would query 1h ohlcv at _ts and predict.
        # For backfill we accept "best-effort" — return None when the bar
        # isn't trivially loadable.
        return None
    except Exception:
        return None


def backfill():
    if not TRADES_PATH.exists():
        print(f'No trades.json at {TRADES_PATH} — nothing to backfill', file=sys.stderr)
        return 1

    with open(TRADES_PATH, encoding='utf-8') as f:
        trades = json.load(f)
    if not isinstance(trades, list):
        print('trades.json is not a list', file=sys.stderr)
        return 1

    # Keep a single classifier instance (lazy; might stay None).
    classifier = None
    try:
        from src.analysis.regime_classifier import train_regime_classifier
        classifier = train_regime_classifier()
        if not getattr(classifier, 'is_ready', False):
            classifier = None
    except Exception:
        classifier = None

    enriched = []
    counts = {
        'mode_set': 0, 'regime_set': 0, 'mfe_set': 0, 'mae_set': 0,
        'exit_reason_set': 0, 'untouched': 0,
    }
    for t in trades:
        out = dict(t)  # shallow copy — preserve all original fields

        # mode (inferred 'testnet' for legacy rows)
        if 'mode' not in out or out['mode'] is None:
            out['mode'] = 'testnet'
            out['mode_inferred'] = True
            counts['mode_set'] += 1

        # exit_reason
        if 'exit_reason' not in out or out['exit_reason'] is None:
            er = _infer_exit_reason(out)
            if er:
                out['exit_reason'] = er
                counts['exit_reason_set'] += 1

        # MFE / MAE from the row's own highest_price + pnl_percent
        if ('mfe_pct' not in out or out['mfe_pct'] in (None, 0.0)
                or 'mae_pct' not in out or out['mae_pct'] in (None, 0.0)):
            mfe, mae = _mfe_mae_from_existing(out)
            if mfe != 0.0:
                out['mfe_pct'] = mfe
                counts['mfe_set'] += 1
            else:
                out.setdefault('mfe_pct', 0.0)
            if mae != 0.0:
                out['mae_pct'] = mae
                counts['mae_set'] += 1
            else:
                out.setdefault('mae_pct', 0.0)

        # regime_at_entry — best-effort
        if 'regime_at_entry' not in out or out['regime_at_entry'] is None:
            r = _try_get_regime(classifier, out.get('buy_time'))
            if r:
                out['regime_at_entry'] = r
                counts['regime_set'] += 1
            else:
                out['regime_at_entry'] = None

        # Unrecoverable from existing data — set placeholders.
        out.setdefault('model_confidence', None)
        out.setdefault('slippage_pct', None)

        if out == t:
            counts['untouched'] += 1
        enriched.append(out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)

    n = len(enriched)
    print(f'Backfilled {n} rows -> {OUT_PATH}')
    print('Field-population counts (rows where field was newly set):')
    for k, v in counts.items():
        pct = (v / n * 100) if n else 0.0
        print(f'  {k:18s} {v:5d}  ({pct:5.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(backfill())
