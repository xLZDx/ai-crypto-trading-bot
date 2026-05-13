# BPMN 2.0 business-process diagrams

Three diagrams, valid BPMN 2.0 XML, renderable by any BPMN tool
(bpmn-js / Camunda Modeler / Draw.io / Signavio / Bizagi):

| File | Process | Swimlanes | Gateways |
|---|---|---|---|
| [TrainingFlow.bpmn](TrainingFlow.bpmn) | Training pipeline — operator click → KEEP/REVIEW/RETIRE | 7 (Operator, Dashboard, KPI Gate pre, MLE pre, Trainer, MLE post, KPI Gate 3-strike) | 4 XOR |
| [TradingFlow.bpmn](TradingFlow.bpmn) | Live trading — one WS tick to filled order | 6 (Market, Signal, Specialists, Risk, Exec, Outcome) | 3 XOR + 2 parallel (fan-out / fan-in) |
| [RegistryClaim.bpmn](RegistryClaim.bpmn) | `process_registry.claim_role` atomic decision tree | 5 (Caller, FileLock, Liveness, Decision, Audit) | 3 XOR |

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
