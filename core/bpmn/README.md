# BPMN 2.0 business-process diagrams

Eight diagrams, one per PNG category in [../diagrams/](../diagrams/). All
valid BPMN 2.0 XML, renderable by bpmn-js / Camunda Modeler / Draw.io /
Signavio / Bizagi:

| File | Process | Swimlanes | Gateways | Matches PNG |
|---|---|---|---|---|
| [TrainingFlow.bpmn](TrainingFlow.bpmn) | Training pipeline — operator → KEEP/REVIEW/RETIRE | 7 | 4 XOR | training_business_flow.png |
| [TradingFlow.bpmn](TradingFlow.bpmn) | Live trading — WS tick → filled order | 6 | 3 XOR + 2 parallel (fan-out/fan-in) | trading_business_flow.png |
| [RegistryClaim.bpmn](RegistryClaim.bpmn) | `process_registry.claim_role` atomic decision tree | 5 | 3 XOR | process_registry.png |
| [ModelsLifecycle.bpmn](ModelsLifecycle.bpmn) | MLPredictor: load → HMAC verify → predict_proba | 6 | 2 XOR | models_hierarchy.png |
| [InfraStartup.bpmn](InfraStartup.bpmn) | `restart_all.ps1` → 11 process roles claimed | 6 | 4 parallel (2 fan-out + 2 fan-in) | infrastructure_topology.png |
| [AgentLifecycle.bpmn](AgentLifecycle.bpmn) | BaseAgent: init → subscribe → loop → stop | 6 | 2 XOR (exception, still running) | agents_class_hierarchy.png |
| [TrainerDispatch.bpmn](TrainerDispatch.bpmn) | get_trainer_agent factory → train → HMAC sign → meta JSON | 8 | 2 XOR (known key, train raised) | trainer_agents_class_hierarchy.png |
| [RiskGates.bpmn](RiskGates.bpmn) | RiskAgent — 9-gate stack traversal (fail-closed) | 8 | **9 XOR (one per gate)** | risk_subsystem.png |

**Total: 68 tasks, 25 XOR gateways, 6 parallel gateways, 145 sequence flows
across 52 swimlanes.**

## View in browser

The bundled viewer uses bpmn-js v17 from unpkg CDN.

```powershell
cd "D:\test 2\AI trading assistance\core\bpmn"
python -m http.server 8088
# open http://localhost:8088/ in any browser
```

Switch between flows with the dropdown. Pan with drag, zoom with scroll,
download as XML or SVG.

## View in Camunda Modeler

```
File → Open File → core/bpmn/TrainingFlow.bpmn
```

Fully editable. Add custom data objects, message flows, etc.

## View in Draw.io

```
File → Open → Device → core/bpmn/TrainingFlow.bpmn
(choose "Import BPMN" when prompted)
```

Draw.io converts the BPMN to its internal mxGraph format. Round-trip is
lossy — keep the .bpmn files as the source of truth.

## Regenerate from source

The diagrams are generated programmatically by
[../../tools/render_bpmn.py](../../tools/render_bpmn.py). Edit the Python
flow definitions there, then:

```bash
venv/Scripts/python.exe tools/render_bpmn.py
```

Each flow is a Python dataclass with `Lane`, `Node`, and `Edge` objects;
layout is auto-computed via a grid placement (column index per node). No
hand-tweaking of XML coordinates needed.

## Why BPMN over Mermaid / ASCII / UML

| Concept | BPMN | Mermaid | ASCII |
|---|---|---|---|
| Swimlanes (zones of responsibility) | first-class `<bpmn:lane>` | none (workaround: subgraphs) | none |
| Parallel gateways (AND-fan-out/fan-in) | first-class `parallelGateway` | none (use `par … and …` in sequence) | none |
| Decision gateways (XOR/OR) | first-class with condition expressions | conditional in `flowchart` | none |
| Industry tool support | universal | GitHub web only | none |
| Edit cycle | round-trip via Modeler | edit Markdown, re-render | edit + reformat |
| Use for business audiences | designed for it | developer-only | developer-only |

The other diagram artefacts in this repo serve different audiences:
- **[../diagrams/*.png](../diagrams/)** — UML class diagrams (engineering)
- **[../UML_CLASS_DIAGRAMS_2026-05-13.md](../UML_CLASS_DIAGRAMS_2026-05-13.md)** — Mermaid class diagrams (engineering, GitHub web)
- **[../ARCHITECTURE_FLOWS.md](../ARCHITECTURE_FLOWS.md)** — Mermaid sequence/flowchart (engineering, deep detail)
- **[../SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md](../SYSTEM_WORKFLOWS_AND_TRAINING_ROADMAP_2026-05-13.md)** — text + Mermaid (engineering, roadmap)
- **This folder** — BPMN 2.0 (business / CEO / investor audience)
