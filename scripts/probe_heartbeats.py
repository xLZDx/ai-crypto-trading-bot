"""One-shot: clean orphan agents from status + probe heartbeat write."""
from __future__ import annotations
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Step 1: remove orphan agents from status
sp = PROJECT_ROOT / 'data' / 'agent_status.json'
d = json.load(open(sp))
orphans = ['ContinuousTrainerAgent', 'StrategySimulatorAgent']
removed = []
for o in orphans:
    if o in d:
        removed.append(o)
        del d[o]
sp.write_text(json.dumps(d, indent=2))
print(f"Removed orphans: {removed}")

# Step 2: probe whether _write_agent_status works for DataAgent name
from src.engine.agents.agent_bus import _write_agent_status
import logging
logging.basicConfig(level=logging.DEBUG)
_write_agent_status('DataAgent_probe', 'idle', 'probe test', 60.0)
d2 = json.load(open(sp))
key = 'DataAgent_probe'
print(f"DataAgent_probe written: {key in d2}")
if key in d2:
    print(f"  last_hb_ts={d2[key]['last_heartbeat_ts']}")
    del d2[key]
    sp.write_text(json.dumps(d2, indent=2))
