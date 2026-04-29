"""
ScenarioManager — selects and classifies training scenarios from historical GZ data.

Scenario types:
  TRENDING_UP    — sustained uptrend (total return > +5% over window)
  TRENDING_DOWN  — sustained downtrend (total return < -5%)
  RANGING        — low-volatility sideways (ADX proxy < threshold)
  VOLATILE       — high-volatility spikes (std > 3× baseline)
  FUNDING_SQUEEZE — extreme positive/negative funding rates
  COMPOSITE      — random mix (mirrors real-world distribution)

Selection strategy (curriculum + weakness-reinforcement):
  1. Start with COMPOSITE to establish a baseline.
  2. After each training cycle, identify which regime has the lowest
     model accuracy and double its sampling weight.
  3. Converge toward uniform coverage once all regimes are balanced.

The manager maintains a JSON catalog at data/sim_scenario_catalog.json
so scenario metadata survives restarts.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CATALOG_PATH = PROJECT_ROOT / "data" / "sim_scenario_catalog.json"

# Scenario type names
SCENARIO_TYPES = [
    "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE",
    "FUNDING_SQUEEZE", "COMPOSITE",
]

# Default sampling weights (will evolve with training feedback)
_DEFAULT_WEIGHTS = {t: 1.0 for t in SCENARIO_TYPES}

# Window sizes (number of bars) used to classify a period
_WINDOW_BARS = {
    "1m":  2_000,   # ~33h of 1m data
    "1h":  500,     # ~3 weeks of 1h data
    "1d":  180,     # 6 months of daily data
}


class ScenarioManager:
    """
    Manages scenario selection for the live-feed simulator.

    Usage::

        mgr = ScenarioManager()
        scenario = mgr.next_scenario(timeframe="1m", model_metrics={})
        # scenario = {"type": "VOLATILE", "symbol": "BTC_USDT",
        #             "timeframe": "1m", "start": datetime(...), "end": datetime(...)}
    """

    # Symbols available in the GZ archive
    _SYMBOLS = [
        "BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT",
        "ADA_USDT", "AVAX_USDT", "ATOM_USDT",
    ]

    def __init__(self):
        self._weights: dict[str, float] = dict(_DEFAULT_WEIGHTS)
        self._catalog: list[dict] = []
        self._load_catalog()

    # ── public ────────────────────────────────────────────────────────────────

    def next_scenario(
        self,
        timeframe: str = "1m",
        model_metrics: dict[str, Any] | None = None,
    ) -> dict:
        """
        Return the next scenario descriptor.

        Args:
            timeframe:     GZ timeframe to use for the scenario.
            model_metrics: dict of {scenario_type: accuracy} from the trainer;
                           used to up-weight scenarios where accuracy is lowest.
        """
        if model_metrics:
            self._update_weights(model_metrics)

        scenario_type = self._weighted_choice()
        symbol = self._pick_symbol(timeframe)
        start, end = self._pick_date_range(symbol, timeframe, scenario_type)

        scenario = {
            "type":      scenario_type,
            "symbol":    symbol,
            "timeframe": timeframe,
            "start":     start.isoformat() if start else None,
            "end":       end.isoformat() if end else None,
            "window_bars": _WINDOW_BARS.get(timeframe, 1000),
        }
        self._catalog.append(scenario)
        self._save_catalog()
        return scenario

    def classify_bars(self, df: pd.DataFrame) -> str:
        """
        Classify a DataFrame of OHLCV bars into a scenario type.
        Used post-hoc to label what kind of market was replayed.
        """
        if df is None or len(df) < 10:
            return "COMPOSITE"

        try:
            returns = df["close"].pct_change().dropna()
            total_return = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
            vol = returns.std()
            baseline_vol = 0.01  # 1% per bar

            # Check funding if present
            if "funding_rate" in df.columns:
                avg_funding = df["funding_rate"].mean()
                if abs(avg_funding) > 0.0005:
                    return "FUNDING_SQUEEZE"

            if vol > baseline_vol * 3:
                return "VOLATILE"
            elif total_return > 0.05:
                return "TRENDING_UP"
            elif total_return < -0.05:
                return "TRENDING_DOWN"
            elif vol < baseline_vol * 0.5:
                return "RANGING"
            else:
                return "COMPOSITE"
        except Exception as exc:
            logger.debug("[ScenarioMgr] classify_bars error: %s", exc)
            return "COMPOSITE"

    def record_result(self, scenario_type: str, accuracy: float) -> None:
        """Feed back training accuracy for a scenario type to update weights."""
        self._update_weights({scenario_type: accuracy})

    # ── private ───────────────────────────────────────────────────────────────

    def _weighted_choice(self) -> str:
        types = list(self._weights.keys())
        weights = [self._weights[t] for t in types]
        return random.choices(types, weights=weights, k=1)[0]

    def _update_weights(self, metrics: dict[str, Any]) -> None:
        """
        Increase weight for scenario types where accuracy is low.
        Weight = 1 / (accuracy + 0.1) — lower accuracy → higher sampling.
        """
        for stype, acc in metrics.items():
            if stype in self._weights:
                self._weights[stype] = 1.0 / (float(acc) + 0.1)
        # Normalise so weights sum to len(SCENARIO_TYPES)
        total = sum(self._weights.values()) or 1.0
        target = float(len(SCENARIO_TYPES))
        for k in self._weights:
            self._weights[k] = self._weights[k] / total * target

    def _pick_symbol(self, timeframe: str) -> str:
        """Pick a symbol that has a GZ file for the given timeframe."""
        available = [
            s for s in self._SYMBOLS
            if (RAW_DIR / f"{s}_{timeframe}.csv.gz").exists()
        ]
        if not available:
            # Fallback: any GZ for this timeframe
            available = [
                p.stem.replace(f"_{timeframe}.csv", "")
                for p in RAW_DIR.glob(f"*_{timeframe}.csv.gz")
            ]
        return random.choice(available) if available else "BTC_USDT"

    def _pick_date_range(
        self, symbol: str, timeframe: str, scenario_type: str
    ) -> tuple[datetime | None, datetime | None]:
        """
        Choose a random date range within the GZ file.
        For VOLATILE/FUNDING_SQUEEZE we bias toward known high-vol periods
        (2020 Mar, 2022 May/Nov) when they fall within the file's range.
        """
        gz_path = RAW_DIR / f"{symbol}_{timeframe}.csv.gz"
        if not gz_path.exists():
            return None, None

        try:
            # Read only first 500 rows to get start date
            first_chunk = pd.read_csv(
                gz_path, compression="gzip",
                nrows=500, index_col=0, parse_dates=True,
            )
            if first_chunk.index.tz is None:
                first_chunk.index = first_chunk.index.tz_localize("UTC")
            file_start = first_chunk.index[0].to_pydatetime()
            # Estimate file end from size
            bar_secs = {"1s": 1, "1m": 60, "1h": 3600, "1d": 86400}.get(timeframe, 60)
            gz_bytes = gz_path.stat().st_size
            est_bars = int(gz_bytes / 15)
            file_end = file_start + timedelta(seconds=est_bars * bar_secs)

            window_bars = _WINDOW_BARS.get(timeframe, 1000)
            window_secs = window_bars * bar_secs
            window_td = timedelta(seconds=window_secs)

            max_start = file_end - window_td
            if max_start <= file_start:
                return file_start, file_end

            # Bias toward volatile periods for specific scenario types
            if scenario_type in ("VOLATILE", "FUNDING_SQUEEZE"):
                biased = self._volatile_periods(file_start, max_start)
                if biased:
                    start = random.choice(biased)
                    return start, start + window_td

            # Random start within valid range
            delta_secs = int((max_start - file_start).total_seconds())
            offset_secs = random.randint(0, max(delta_secs, 1))
            start = file_start + timedelta(seconds=offset_secs)
            return start, start + window_td

        except Exception as exc:
            logger.warning("[ScenarioMgr] _pick_date_range error: %s", exc)
            return None, None

    @staticmethod
    def _volatile_periods(
        file_start: datetime, file_end: datetime
    ) -> list[datetime]:
        """Known high-volatility crypto periods for biased scenario sampling."""
        candidates = [
            datetime(2020, 3, 12, tzinfo=timezone.utc),  # COVID crash
            datetime(2021, 5, 19, tzinfo=timezone.utc),  # China ban selloff
            datetime(2022, 5, 9,  tzinfo=timezone.utc),  # LUNA collapse
            datetime(2022, 11, 8, tzinfo=timezone.utc),  # FTX collapse
            datetime(2023, 3, 10, tzinfo=timezone.utc),  # Banking crisis
            datetime(2024, 1, 10, tzinfo=timezone.utc),  # BTC ETF approval
            datetime(2025, 2, 3,  tzinfo=timezone.utc),  # DeepSeek shock
        ]
        return [d for d in candidates if file_start <= d <= file_end]

    def _load_catalog(self) -> None:
        try:
            if CATALOG_PATH.exists():
                self._catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            self._catalog = []

    def _save_catalog(self) -> None:
        try:
            CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CATALOG_PATH.write_text(
                json.dumps(self._catalog[-500:], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("[ScenarioMgr] catalog save error: %s", exc)
