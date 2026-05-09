"""Service orchestration layer — process supervision, checkpointing,
file-based pub-sub topics, and the top-level master agent.

Modules in this package:
  - process_health     (Layer 1) → src/utils/process_health.py
  - topics             (Layer 6) → file-based pub-sub
  - checkpoint_writer  (Layer 2, pending) → per-epoch trainer state
  - training_supervisor (Layer 3, pending) → daemon for training jobs
  - backtest_supervisor (Layer 4, pending) → daemon for backtest jobs
  - master_agent       (Layer 5, pending) → top-level supervisor

See ROADMAP_2026_05_10_orchestration.md for the multi-layer plan.
"""
