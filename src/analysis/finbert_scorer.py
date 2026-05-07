"""
finbert_scorer — drop-in upgrade for the crude lexicon `_crude_tone`
used by the news scrapers.

Phase B of the institutional roadmap. Today every GDELT / Reddit /
CryptoCompare backfill scores headlines via a 30-word `_NEG_WORDS /
_POS_WORDS` set. This module replaces that with a real model:

  Primary:   ElKulako/cryptobert  (crypto-domain BERT, fine-tuned on
                                   crypto news + Reddit posts; outputs
                                   bullish/bearish/neutral)
  Fallback:  ProsusAI/finbert     (general financial-news BERT)
  Final:     `_crude_tone`        (existing lexicon — used when neither
                                   model can be loaded, e.g. no GPU /
                                   transformers not installed)

The output is normalised to the same [-1, +1] scale the existing
`tone` parquet column carries, so the downstream news-sentiment loader
in feature_engineering doesn't need to change. Existing partitions
written with crude tone keep working; new partitions written by
re-running the scrapers carry the FinBERT score.

Public surface:
  score_one(text)         -> float    # cached per-text
  score_batch([t1, t2…])  -> [float, …]
  get_active_model()      -> str      # which backend was loaded
  is_ready()              -> bool

Scrapers can opt in via the `tone_model` constructor arg (default:
'auto' — try cryptobert, fall back to finbert, fall back to lexicon).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

# Module-level singleton — loading a 400MB model per scraper call would
# defeat the point. Lazy-init on first use.
_classifier: Callable[[str | list[str]], list] | None = None
_active_model: str = "none"
_load_attempted = False
_load_error: str = ""


# Crypto-news lexicon — same as the scrapers' `_NEG_WORDS / _POS_WORDS`
# but kept here too as the no-deps fallback.
_NEG = {
    "ban", "banned", "crash", "drop", "plunge", "hack", "hacked", "exploit",
    "scam", "fraud", "rug", "rugpull", "bear", "bearish", "lawsuit",
    "investigation", "indicted", "arrested", "down", "loss", "warning",
    "fail", "fails", "failed", "collapsed", "collapse", "selloff", "dump",
    "fud", "regulation", "regulator", "fine", "fined", "delist", "delisted",
}
_POS = {
    "surge", "surges", "rally", "rallies", "soar", "soars", "bull", "bullish",
    "breakout", "ath", "all-time", "approved", "approval", "etf", "adoption",
    "partnership", "launch", "launched", "milestone", "growth", "rises", "rise",
    "gains", "gain", "up", "high", "record", "boost", "boosts",
}


def _lexicon_score(text: str) -> float:
    if not text:
        return 0.0
    words = {w.strip(".,!?:;'\"()").lower() for w in text.split()}
    n = sum(1 for w in words if w in _NEG)
    p = sum(1 for w in words if w in _POS)
    if not (n or p):
        return 0.0
    return round((p - n) / max(p + n, 1), 3)


def _try_load(model_id: str) -> Callable | None:
    """Try loading a HuggingFace pipeline. Returns the pipeline or None.
    Cache to D: drive so the model file doesn't end up on C: (per
    CLAUDE.md). Return None on any error — caller should fall through."""
    try:
        from transformers import pipeline
    except Exception as exc:
        logger.info("[finbert_scorer] transformers not installed: %s", exc)
        return None
    try:
        # Use D:/data/cache/huggingface so we don't fill C:
        cache_dir = os.environ.get("HF_HOME") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "cache", "huggingface"
        )
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        clf = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=-1,           # CPU for inference cheapness
            truncation=True,
            max_length=256,
        )
        # Warm test
        _ = clf("Bitcoin pumps to a new all-time high")
        return clf
    except Exception as exc:
        logger.info("[finbert_scorer] load failed for %s: %s", model_id, exc)
        return None


def _ensure_loaded(preferred: str = "auto") -> None:
    """Lazy-init the singleton. Order: cryptobert → finbert → lexicon.
    `preferred='lexicon'` skips the model entirely (testing convenience)."""
    global _classifier, _active_model, _load_attempted, _load_error
    if _load_attempted:
        return
    _load_attempted = True
    if preferred == "lexicon":
        _classifier = None
        _active_model = "lexicon"
        return
    candidates = []
    if preferred in ("auto", "cryptobert"):
        candidates.append(("cryptobert", "ElKulako/cryptobert"))
    if preferred in ("auto", "finbert"):
        candidates.append(("finbert", "ProsusAI/finbert"))
    for name, mid in candidates:
        clf = _try_load(mid)
        if clf is not None:
            _classifier = clf
            _active_model = name
            logger.info("[finbert_scorer] active model: %s (%s)", name, mid)
            return
    _classifier = None
    _active_model = "lexicon"
    _load_error = "no model could be loaded — using lexicon fallback"


def _label_to_score(label: str, conf: float) -> float:
    """Map (label, confidence) → tone in [-1, +1]. Both finbert and
    cryptobert use 'bullish/bearish/neutral' or 'positive/negative/
    neutral' — handle both. Confidence already scales the magnitude."""
    l = (label or "").lower()
    if "bull" in l or "pos" in l or "label_2" in l:
        return round(float(conf), 3)
    if "bear" in l or "neg" in l or "label_0" in l:
        return -round(float(conf), 3)
    return 0.0


@lru_cache(maxsize=10_000)
def score_one(text: str, preferred: str = "auto") -> float:
    """Score one headline. Cached so re-scoring identical headlines (eg
    the same article appearing in multiple coin buckets) is free."""
    _ensure_loaded(preferred)
    if _classifier is None:
        return _lexicon_score(text or "")
    if not text:
        return 0.0
    try:
        out = _classifier(text)
        if isinstance(out, list) and out:
            r = out[0]
            return _label_to_score(r.get("label", ""), r.get("score", 0.0))
    except Exception as exc:
        logger.debug("[finbert_scorer] inference err on %s: %s", text[:60], exc)
    return _lexicon_score(text)


def score_batch(texts: Iterable[str],
                preferred: str = "auto",
                batch_size: int = 32) -> list[float]:
    """Score many headlines. Far faster than score_one in a loop because
    transformers batches the forward pass. Falls back to score_one (which
    is lru_cached) when the model isn't available."""
    _ensure_loaded(preferred)
    texts = list(texts)
    if _classifier is None:
        return [_lexicon_score(t or "") for t in texts]
    if not texts:
        return []
    try:
        out = _classifier(texts, batch_size=batch_size)
    except Exception as exc:
        logger.debug("[finbert_scorer] batch err: %s — falling back per-item", exc)
        return [score_one(t, preferred) for t in texts]
    return [_label_to_score(r.get("label", ""), r.get("score", 0.0)) for r in out]


def get_active_model() -> str:
    return _active_model


def is_ready() -> bool:
    """True iff a real model is loaded (not the lexicon fallback)."""
    _ensure_loaded("auto")
    return _classifier is not None


__all__ = [
    "score_one", "score_batch",
    "get_active_model", "is_ready",
]
