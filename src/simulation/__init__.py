"""
Simulation package — live-feed simulator for training models that require
real-time data (Scalping 1m, TFT Market-Maker, OU filter).

Replays 10+ years of historical GZ data as a synthetic live feed so every
strategy and model can be trained and paper-traded without waiting for real
market time to pass.

Architecture:
  MarketReplay       — streams bars from data/raw/*.csv.gz lazily
  ScenarioManager    — selects & classifies training scenarios
  SimulatorDataStore — DuckDB store for paper trades & training metrics
  SimulatorAgent     — AgentBus agent that orchestrates the replay
  ContinuousTrainerAgent — subscribes to sim_candle and trains models online
"""
from src.simulation.data_store import SimulatorDataStore
from src.simulation.market_replay import MarketReplay
from src.simulation.scenario_manager import ScenarioManager

__all__ = ["SimulatorDataStore", "MarketReplay", "ScenarioManager"]
