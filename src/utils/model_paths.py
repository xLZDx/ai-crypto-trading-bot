"""
model_paths — single source of truth for where ML model artifacts live on
disk. Resolves the path for any (model_key, timeframe) pair.

Why this exists:
The codebase historically used hardcoded names like models/btc_rf_model.joblib
(base @ 1h), models/scalping_model.joblib (scalping @ 1m), etc. With the
multi-timeframe initiative we now want to train the same model at 5m, 1h,
4h, 1d, 1w, 1mo and compare stability — so we need a per-TF naming
convention WITHOUT breaking the bot's inference path that loads the legacy
files today.

The convention:
  - Per-TF artifact:  models/<key>_<tf>_model.<ext>  + models/<key>_<tf>_meta.json
  - Legacy fallback:  models/<legacy_name>           + models/<legacy_meta>
                      (used by inference and any caller that doesn't pass
                      a timeframe — i.e. the historical default)

The trainer also writes the legacy file when its TF matches the historical
canonical TF (1h for base/trend/futures, 1m for scalping), so the bot's
inference engine keeps loading the same path while the new TF variants
pile up alongside.

Public surface:
  KEYS                                  — frozenset of valid model keys
  CANONICAL_TF[key]                     — the legacy default TF per key
  artifact_paths(key, tf)               — {model, meta, legacy_model, legacy_meta}
  resolve_model_for_inference(key, tf)  — pick the best on-disk file to load
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# Eight model families currently in the pipeline. Mirrors _ML / _MODEL_FILES
# in src/dashboard/app.py — keep these in sync.
KEYS = frozenset({
    "base", "trend", "futures", "scalping", "tft", "oft", "meta", "regime",
})

# Canonical timeframe per model — the TF the trainer historically defaults
# to and the TF the bot's inference engine loads via legacy file names.
# The trainer writes BOTH the per-TF and the legacy file when invoked at
# the canonical TF, so the bot stays compatible.
CANONICAL_TF: dict[str, str] = {
    "base":     "1h",
    "trend":    "1h",
    "futures":  "1h",
    "scalping": "1m",
    "tft":      "1h",
    "oft":      "1m",     # OFT is microstructure / orderbook, not bars
    "meta":     "1h",     # meta-labeler aggregates across timeframes too
    "regime":   "1h",
}

# Legacy names — exactly what the bot's inference engine + dashboard load
# today. We never break these; the per-TF names live alongside.
LEGACY_MODEL_NAME: dict[str, str] = {
    "base":     "btc_rf_model.joblib",
    "trend":    "trend_model.joblib",
    "futures":  "futures_short_model.joblib",
    "scalping": "scalping_model.joblib",
    "tft":      "tft_model.pt",
    "oft":      "oft_model.pt",
    "meta":     "meta_labeler.joblib",
    "regime":   "regime_classifier.joblib",
}
LEGACY_META_NAME: dict[str, str] = {
    "base":     "btc_rf_model_meta.json",
    "trend":    "trend_model_meta.json",
    "futures":  "futures_short_model_meta.json",
    "scalping": "scalping_model_meta.json",
    "tft":      "tft_model_meta.json",
    "oft":      "oft_model_meta.json",
    "meta":     "meta_labeler_meta.json",
    "regime":   "regime_classifier_meta.json",
}

# Whether each model serialises as joblib (sklearn) or torch (.pt).
_EXT: dict[str, str] = {
    "base":     ".joblib",
    "trend":    ".joblib",
    "futures":  ".joblib",
    "scalping": ".joblib",
    "tft":      ".pt",
    "oft":      ".pt",
    "meta":     ".joblib",
    "regime":   ".joblib",
}


def _check_key(key: str) -> None:
    if key not in KEYS:
        raise ValueError(f"unknown model key {key!r}; valid: {sorted(KEYS)}")


def per_tf_model_name(key: str, tf: str) -> str:
    """Return the per-TF artifact filename for a model.
    Example: per_tf_model_name('base', '4h') == 'base_4h_model.joblib'."""
    _check_key(key)
    return f"{key}_{tf}_model{_EXT[key]}"


def per_tf_meta_name(key: str, tf: str) -> str:
    _check_key(key)
    return f"{key}_{tf}_meta.json"


def artifact_paths(key: str, tf: str) -> dict[str, Path]:
    """Return all four paths the trainer should consider for a (key, tf):
       - model:        per-TF artifact ALWAYS written
       - meta:         per-TF JSON ALWAYS written
       - legacy_model: legacy filename, written ONLY when tf == CANONICAL_TF
       - legacy_meta:  legacy meta file, same condition
    Callers that aren't multi-TF aware load from legacy_*.
    """
    _check_key(key)
    return {
        "model":        MODELS_DIR / per_tf_model_name(key, tf),
        "meta":         MODELS_DIR / per_tf_meta_name(key, tf),
        "legacy_model": MODELS_DIR / LEGACY_MODEL_NAME[key],
        "legacy_meta":  MODELS_DIR / LEGACY_META_NAME[key],
        "is_canonical": tf == CANONICAL_TF[key],
    }


def resolve_model_for_inference(key: str, tf: str | None = None) -> Path | None:
    """Pick the file the inference engine should load for (key, tf).
    Resolution order:
      1. Per-TF file if it exists (e.g. base_4h_model.joblib)
      2. Legacy file (always exists for trained models)
      3. None if neither
    Pass tf=None to always go straight to legacy."""
    _check_key(key)
    if tf is not None and tf != CANONICAL_TF[key]:
        per_tf = MODELS_DIR / per_tf_model_name(key, tf)
        if per_tf.exists():
            return per_tf
    legacy = MODELS_DIR / LEGACY_MODEL_NAME[key]
    if legacy.exists():
        return legacy
    return None


def resolve_meta_for_inference(key: str, tf: str | None = None) -> Path | None:
    """Same as resolve_model_for_inference but for the meta JSON."""
    _check_key(key)
    if tf is not None and tf != CANONICAL_TF[key]:
        per_tf = MODELS_DIR / per_tf_meta_name(key, tf)
        if per_tf.exists():
            return per_tf
    legacy = MODELS_DIR / LEGACY_META_NAME[key]
    if legacy.exists():
        return legacy
    return None


def list_per_tf_artifacts(key: str) -> list[tuple[str, Path, Path]]:
    """Scan models/ and return [(tf, model_path, meta_path)] for every
    per-TF artifact present for this key. Used by the dashboard to enumerate
    multi-TF model variants without hardcoding the TF list."""
    _check_key(key)
    out: list[tuple[str, Path, Path]] = []
    if not MODELS_DIR.exists():
        return out
    ext = _EXT[key]
    prefix = f"{key}_"
    suffix = f"_model{ext}"
    for p in MODELS_DIR.iterdir():
        if not p.is_file():
            continue
        n = p.name
        if not (n.startswith(prefix) and n.endswith(suffix)):
            continue
        tf = n[len(prefix):-len(suffix)]
        if not tf:
            continue
        meta = MODELS_DIR / f"{key}_{tf}_meta.json"
        out.append((tf, p, meta))
    return sorted(out, key=lambda r: r[0])


# CLI smoke: `python -m src.utils.model_paths base 4h`
if __name__ == "__main__":
    import json as _json, sys
    key = sys.argv[1] if len(sys.argv) > 1 else "base"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "1h"
    paths = artifact_paths(key, tf)
    paths = {k: str(v) for k, v in paths.items()}
    paths["resolved_model"] = str(resolve_model_for_inference(key, tf))
    paths["resolved_meta"]  = str(resolve_meta_for_inference(key, tf))
    paths["per_tf_present"] = {tf: (str(m), str(meta))
                               for tf, m, meta in list_per_tf_artifacts(key)}
    print(_json.dumps(paths, indent=2))
