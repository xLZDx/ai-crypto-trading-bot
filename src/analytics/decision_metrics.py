"""
DecisionMetrics — pre-aggregated summaries used to GO / NO-GO trades.

Five buckets:
  • coverage    — how complete is data for (symbol, timeframe, period)?
  • feature     — current feature/regime/sentiment snapshot
  • model_health — is each loaded model fresh + accurate enough?
  • risk        — open beta exposure, drawdown, circuit-breaker state
  • execution   — latency, slippage budget remaining

Returned as `DecisionSummary` namedtuple — directly serializable to JSON
for the dashboard `/api/decision_summary` endpoint and the live bot's
"can I trade now?" check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class DecisionSummary:
    symbol:        str
    timeframe:     str
    coverage:      dict = field(default_factory=dict)
    feature:       dict = field(default_factory=dict)
    model_health:  dict = field(default_factory=dict)
    risk:          dict = field(default_factory=dict)
    execution:     dict = field(default_factory=dict)
    go:            bool = True
    blockers:      list = field(default_factory=list)
    as_of:         str  = ""

    def to_dict(self):
        return asdict(self)


class DecisionMetrics:
    """Wraps DataLens + the loaded models + risk state."""

    def __init__(self, lens=None):
        if lens is None:
            from src.analytics.data_lens import DataLens
            lens = DataLens()
        self.lens = lens

    # ── Bucket builders ────────────────────────────────────────────────────

    def coverage(self, *, symbol: str, timeframe: str) -> dict:
        try:
            st = self.lens._p().symbol_status(symbol, timeframe=timeframe)
            return {
                "rows":       st.rows,
                "earliest":   st.earliest,
                "latest":     st.latest,
                "size_bytes": st.size_bytes,
                "partitions": st.partitions,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def feature(self, *, symbol: str, timeframe: str) -> dict:
        try:
            from datetime import timedelta
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=7)
            df = self.lens.training_frame(
                symbol=symbol, timeframe=timeframe,
                start=start, end=end,
            )
            if df is None or df.empty:
                return {"error": "no recent data"}
            last = df.iloc[-1].to_dict()
            return {
                "close":              float(last.get("close", 0)),
                "volume":             float(last.get("volume", 0)),
                "funding_rate":       float(last.get("funding_rate", 0) or 0),
                "news_sentiment_24h": float(last.get("news_sentiment_24h", 0) or 0),
                "as_of":              str(last.get("timestamp", "")),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def model_health(self) -> dict:
        out = {}
        from pathlib import Path
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("regime_classifier", "oft_model",
                     "btc_rf_model", "scalping_model",
                     "trend_model", "meta_labeler"):
            for ext in (".joblib", ".pt"):
                p = models_dir / f"{name}{ext}"
                if p.exists():
                    out[name] = {
                        "exists":     True,
                        "size_bytes": p.stat().st_size,
                        "mtime":      datetime.fromtimestamp(
                                          p.stat().st_mtime, tz=timezone.utc).isoformat(),
                    }
                    break
            else:
                out[name] = {"exists": False}
        return out

    def risk(self) -> dict:
        # Pull from `data/state.json` (live bot state)
        try:
            from src.utils.safe_json import read_json
            from pathlib import Path
            state = read_json(str(Path(__file__).resolve().parents[2] /
                                   "data" / "state.json")) or {}
            return {
                "drawdown_pct":   float(state.get("drawdown_pct", 0) or 0),
                "open_positions": int(state.get("open_positions", 0) or 0),
                "circuit_breaker": bool(state.get("circuit_breaker", False)),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def execution(self) -> dict:
        from src.data_ingestion.rate_limiter import stats
        return {"rate_limiter": stats()}

    # ── Top-level GO/NO-GO ─────────────────────────────────────────────────

    def summarize(self, *, symbol: str, timeframe: str) -> DecisionSummary:
        cov  = self.coverage(symbol=symbol, timeframe=timeframe)
        feat = self.feature(symbol=symbol, timeframe=timeframe)
        mh   = self.model_health()
        risk = self.risk()
        ex   = self.execution()

        blockers = []
        # Hard gates
        if not cov.get("rows"):
            blockers.append("no parquet rows for this symbol/timeframe")
        if mh.get("regime_classifier", {}).get("exists") is False:
            blockers.append("regime_classifier not trained")
        if risk.get("circuit_breaker"):
            blockers.append("circuit breaker active")
        if (risk.get("drawdown_pct") or 0) > 5:
            blockers.append(f"drawdown {risk['drawdown_pct']:.2f}% > 5%")

        return DecisionSummary(
            symbol=symbol, timeframe=timeframe,
            coverage=cov, feature=feat,
            model_health=mh, risk=risk, execution=ex,
            go=not blockers,
            blockers=blockers,
            as_of=datetime.now(timezone.utc).isoformat(),
        )


__all__ = ["DecisionMetrics", "DecisionSummary"]
