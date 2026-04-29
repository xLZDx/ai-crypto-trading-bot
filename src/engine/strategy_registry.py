"""
Strategy Registry — single source of truth for all strategies and models.

Every strategy that exists in either the live bot or the backtester is declared
here. Both systems read this registry to decide what to run and whether it is
enabled. Enabling/disabling is persisted in data/strategy_config.json so the
dashboard can control it at runtime.

Adding a new strategy:
  1. Add an entry to REGISTRY below.
  2. Implement it in the live bot (main.py) and/or backtester (backtester.py).
  3. Set can_live / can_backtest based on what you actually wired.
  4. The sync card on the dashboard will automatically show its status.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "data" / "strategy_config.json"

# ── Registry ──────────────────────────────────────────────────────────────────
# can_live:      strategy is implemented in the live bot (main.py)
# can_backtest:  strategy is implemented in the backtester (backtester.py)
# models:        model files required (empty = rules-only)
# signal_col:    column name used in the backtester DataFrame
REGISTRY: dict[str, dict[str, Any]] = {
    # ── Group A: original rule-based ─────────────────────────────────────────
    "RSI_MeanReversion": {
        "label": "RSI Mean Reversion",
        "description": "Long RSI<30, short RSI>70 on 1h bars",
        "group": "A_Original",
        "signal_col": "signal_rsi",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "MACD_Momentum": {
        "label": "MACD Momentum",
        "description": "Long when MACD histogram > 0, short when < 0",
        "group": "A_Original",
        "signal_col": "signal_macd",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "BB_Reversion": {
        "label": "Bollinger Band Reversion",
        "description": "Long at lower band (BB%B < 0.1), short at upper (>0.9)",
        "group": "A_Original",
        "signal_col": "signal_bb",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Ensemble_A": {
        "label": "Ensemble A (RSI+MACD+BB)",
        "description": "Average of RSI, MACD, BB signals",
        "group": "A_Original",
        "signal_col": "signal_ensemble",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },

    # ── Group B: new rule-based ───────────────────────────────────────────────
    "VWAP_Reversion": {
        "label": "VWAP Reversion",
        "description": "Long when price 0.5% below VWAP, short when 0.5% above",
        "group": "B_New",
        "signal_col": "signal_vwap",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Donchian_Breakout": {
        "label": "Donchian Breakout",
        "description": "Long on 20-bar channel high break, short on low break",
        "group": "B_New",
        "signal_col": "signal_donchian",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Keltner_Breakout": {
        "label": "Keltner Channel Breakout",
        "description": "Long/short when price exits Keltner channel",
        "group": "B_New",
        "signal_col": "signal_keltner",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Funding_Arb": {
        "label": "Funding Rate Arb",
        "description": "Short on positive funding (>0.1%), long on negative (<-0.05%)",
        "group": "B_New",
        "signal_col": "signal_funding",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "OFI_Momentum": {
        "label": "OFI Momentum",
        "description": "Long on high buy-side order flow imbalance (OFI-Z > 1.5)",
        "group": "B_New",
        "signal_col": "signal_ofi",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Volatility_Breakout": {
        "label": "Volatility Breakout (TTM Squeeze)",
        "description": "BB inside Keltner squeeze → breakout with volume surge",
        "group": "B_New",
        "signal_col": "signal_vol_breakout",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Ensemble_B": {
        "label": "Ensemble B (VWAP+Don+Kelt+Fund+OFI)",
        "description": "Average of all Group B signals",
        "group": "B_New",
        "signal_col": "signal_ensemble_b",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },

    # ── ML model signals ──────────────────────────────────────────────────────
    "ElliottWave_ML": {
        "label": "Elliott Wave + Base ML",
        "description": "Live: Elliott Wave pivots + RF model. BT: momentum proxy (5-bar mom + SMA50 + ML)",
        "group": "ML",
        "signal_col": "signal_elliott_proxy",
        "models": ["btc_rf_model.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "Base_ML": {
        "label": "Base RF Model",
        "description": "Random Forest on 1h OHLCV + sentiment features",
        "group": "ML",
        "signal_col": "signal_base_ml",
        "models": ["btc_rf_model.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "Trend_ML": {
        "label": "Trend RF Model",
        "description": "RF model on Donchian+Keltner+ADX trend features",
        "group": "ML",
        "signal_col": "signal_trend_ml",
        "models": ["trend_model.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "Futures_Short_ML": {
        "label": "Futures Short RF Model",
        "description": "RF model for futures shorting opportunities",
        "group": "ML",
        "signal_col": "signal_futures_ml",
        "models": ["futures_short_model.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "Scalping_ML": {
        "label": "Scalping RF Model (1m)",
        "description": "1-minute RF model for scalping",
        "group": "ML",
        "signal_col": "signal_scalping",
        "models": ["scalping_model.joblib"],
        "can_live": True,
        "can_backtest": False,  # 1m data not in hourly backtest
    },
    "TFT_MarketMaker": {
        "label": "TFT + Avellaneda-Stoikov MM",
        "description": "Neural TFT forecast feeds market-making spread optimizer",
        "group": "ML",
        "signal_col": "signal_tft_mm",
        "models": ["tft_model.pt"],
        "can_live": True,
        "can_backtest": False,  # market-making needs live order book
    },

    # ── Filters / overlays ───────────────────────────────────────────────────
    "MetaLabeler_Filter": {
        "label": "Meta-Labeler Filter",
        "description": "HistGBT classifier filters low-quality signals before order submission",
        "group": "Filter",
        "signal_col": "meta_filter",
        "models": ["meta_labeler.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "RegimeClassifier_Router": {
        "label": "Regime Classifier Router",
        "description": "GMM routes RANGING/TRENDING/VOLATILE → adjusts strategy selection and size",
        "group": "Filter",
        "signal_col": "regime",
        "models": ["regime_classifier.joblib"],
        "can_live": True,
        "can_backtest": True,
    },
    "GARCH_PositionSizing": {
        "label": "GARCH Volatility Position Sizing",
        "description": "Live: GARCH vol spike halves size. BT: 5-bar realized vol vs 60-bar mean proxy",
        "group": "RiskFilter",
        "signal_col": "garch_size_mult",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "MTF_SMA200_Filter": {
        "label": "MTF SMA-200 Filter",
        "description": "Blocks trades against the 1-day SMA-200 macro trend",
        "group": "RiskFilter",
        "signal_col": "signal_mtf_filter",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "OU_MeanReversion_Filter": {
        "label": "Ornstein-Uhlenbeck Filter",
        "description": "Blocks trend entries when price is statistically stretched (>2σ from OU mean)",
        "group": "RiskFilter",
        "signal_col": "ou_filter",
        "models": [],
        "can_live": True,
        "can_backtest": False,  # requires online calibration
    },

    # ── New high-performance strategies ──────────────────────────────────────
    "Ichimoku_Cloud": {
        "label": "Ichimoku Cloud",
        "description": "TK cross above/below cloud + Chikou confirmation — trend-following on 1h bars",
        "group": "B_New",
        "signal_col": "signal_ichimoku",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "Supertrend": {
        "label": "SuperTrend (ATR×3, 10)",
        "description": "ATR trailing stop flips; emits signal only on direction change — strong momentum filter",
        "group": "B_New",
        "signal_col": "signal_supertrend",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "MACD_Divergence": {
        "label": "MACD Centerline Cross + Divergence",
        "description": "Centerline crossovers and price/MACD divergence — higher-quality MACD signals",
        "group": "B_New",
        "signal_col": "signal_macd_div",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
    "OU_Entry": {
        "label": "OU Mean-Reversion Entry",
        "description": "Price >2σ from OU mean in RANGING regime → fade the deviation (primary signal)",
        "group": "B_New",
        "signal_col": "signal_ou_entry",
        "models": [],
        "can_live": True,
        "can_backtest": True,
    },
}

# ── Default enabled state ──────────────────────────────────────────────────────
_DEFAULTS: dict[str, dict[str, bool]] = {
    name: {
        "live":      info["can_live"],
        "backtest":  info["can_backtest"],
    }
    for name, info in REGISTRY.items()
}


# ── Config I/O ────────────────────────────────────────────────────────────────

def load_config() -> dict[str, dict[str, bool]]:
    """Load strategy_config.json, merging in any new registry entries."""
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: add new registry entries with defaults, keep user values for existing
            merged = {**_DEFAULTS, **{k: v for k, v in saved.items() if k in REGISTRY}}
            # Write back if new strategies were added
            if set(merged.keys()) != set(saved.keys()):
                _write_config(merged)
            return merged
    except Exception as e:
        logger.warning("Could not load strategy_config.json: %s — using defaults", e)
    return dict(_DEFAULTS)


def _write_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_config(config: dict[str, dict[str, bool]]) -> None:
    """Persist the full config dict."""
    # Only save keys present in the registry
    clean = {k: v for k, v in config.items() if k in REGISTRY}
    _write_config(clean)


def update_strategy(name: str, live: bool | None = None, backtest: bool | None = None) -> dict:
    """Toggle a single strategy's live/backtest flag and persist."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}")
    cfg = load_config()
    entry = cfg.get(name, dict(_DEFAULTS.get(name, {"live": False, "backtest": False})))
    if live is not None:
        # Only allow enabling if the registry says it can_live / can_backtest
        entry["live"]     = bool(live)     if REGISTRY[name]["can_live"]     else False
        entry["backtest"] = bool(backtest) if REGISTRY[name]["can_backtest"] else False
    if backtest is not None:
        entry["backtest"] = bool(backtest) if REGISTRY[name]["can_backtest"] else False
    cfg[name] = entry
    save_config(cfg)
    return entry


# ── Sync report ───────────────────────────────────────────────────────────────

def get_sync_report() -> dict:
    """
    Returns a full sync status dict consumed by /api/strategy-sync.
    Each entry has: label, group, description, can_live, can_backtest,
    live_enabled, backtest_enabled, sync_status, models_present.
    """
    cfg = load_config()
    models_dir = PROJECT_ROOT / "models"

    entries = []
    for name, info in REGISTRY.items():
        entry_cfg = cfg.get(name, {"live": info["can_live"], "backtest": info["can_backtest"]})
        live_en   = entry_cfg.get("live",     False)
        bt_en     = entry_cfg.get("backtest", False)

        models_ok = all((models_dir / m).exists() for m in info["models"]) if info["models"] else True
        models_present = {m: (models_dir / m).exists() for m in info["models"]}

        # Sync status
        if info["can_live"] and info["can_backtest"]:
            if live_en and bt_en:
                sync_status = "synced"
            elif live_en and not bt_en:
                sync_status = "live_only"
            elif bt_en and not live_en:
                sync_status = "backtest_only"
            else:
                sync_status = "disabled"
        elif info["can_live"] and not info["can_backtest"]:
            sync_status = "live_only_by_design"
        elif info["can_backtest"] and not info["can_live"]:
            sync_status = "backtest_only_by_design"
        else:
            sync_status = "disabled"

        entries.append({
            "name":            name,
            "label":           info["label"],
            "group":           info["group"],
            "description":     info["description"],
            "can_live":        info["can_live"],
            "can_backtest":    info["can_backtest"],
            "live_enabled":    live_en,
            "backtest_enabled": bt_en,
            "sync_status":     sync_status,
            "models":          info["models"],
            "models_ok":       models_ok,
            "models_present":  models_present,
        })

    # Summary counts
    total   = len(entries)
    synced  = sum(1 for e in entries if e["sync_status"] == "synced")
    gaps    = sum(1 for e in entries if e["sync_status"] in ("live_only", "backtest_only"))
    missing = sum(1 for e in entries if not e["models_ok"])

    return {
        "strategies": entries,
        "summary": {
            "total":   total,
            "synced":  synced,
            "gaps":    gaps,
            "missing_models": missing,
        },
    }


# ── Convenience helpers used by live bot and backtester ───────────────────────

def is_enabled_live(name: str) -> bool:
    return load_config().get(name, {}).get("live", False)


def is_enabled_backtest(name: str) -> bool:
    return load_config().get(name, {}).get("backtest", False)


def enabled_backtest_signal_cols() -> list[tuple[str, str, str]]:
    """Return [(strategy_name, label, signal_col)] for all backtest-enabled strategies."""
    cfg = load_config()
    return [
        (name, info["label"], info["signal_col"])
        for name, info in REGISTRY.items()
        if cfg.get(name, {}).get("backtest", False) and info["can_backtest"]
    ]
