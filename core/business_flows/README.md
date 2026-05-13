# Business-flow PNGs — presentation-ready

Eight presentation-quality PNGs, one per business process, in BPMN visual
style. Drop straight into slides — no browser, no viewer, no install.

| File | Process | Best slide use |
|---|---|---|
| [TrainingFlow.png](TrainingFlow.png) | Training pipeline — operator click → KEEP/REVIEW/RETIRE | "How the bot learns + when models retire" |
| [TradingFlow.png](TradingFlow.png) | Live trading — WS tick → filled order | "What happens between a market tick and an order" |
| [RiskGates.png](RiskGates.png) | RiskAgent 9-gate stack | "How risk is enforced (9 gates, fail-closed)" |
| [ModelsLifecycle.png](ModelsLifecycle.png) | MLPredictor load + HMAC + predict | "Model integrity + inference path" |
| [RegistryClaim.png](RegistryClaim.png) | `process_registry.claim_role` decision tree | "How duplicate-process is prevented" |
| [AgentLifecycle.png](AgentLifecycle.png) | BaseAgent thread lifecycle | "How an agent boots, runs, and stops" |
| [TrainerDispatch.png](TrainerDispatch.png) | Factory → train → HMAC sign → meta JSON | "How a training job becomes a signed model" |
| [InfraStartup.png](InfraStartup.png) | `restart_all.ps1` → 11 process roles | "Boot sequence (parallel spawn)" |

## Style

Draw.io BPMN palette:
- Lanes — alternating white / light-gray stripes
- Lane label — dark blue band on left with rotated white text
- Tasks — rounded light-blue rectangles (`#dae8fc` / `#6c8ebf`)
- Start events — green-edged circles
- End events — red-edged circles (thick border)
- XOR gateways — yellow diamonds with × inside
- Parallel gateways — yellow diamonds with + inside
- Sequence flows — orthogonal arrows with optional labels

200 DPI, white background, tight margins. Average size 1.5–3 MB per PNG.

## Regenerate

```bash
venv/Scripts/python.exe tools/render_business_flow_pngs.py
```

The renderer ([../../tools/render_business_flow_pngs.py](../../tools/render_business_flow_pngs.py))
imports `Flow` definitions from [../../tools/render_bpmn.py](../../tools/render_bpmn.py)
— edit one source, both PNG and BPMN regenerate consistently.

## When to use what

- **Slide deck / one-pager / hand-out** → these PNGs.
- **Editable BPMN that imports into Camunda Modeler / Draw.io** → [../bpmn/*.bpmn](../bpmn/) (same content, BPMN 2.0 XML).
- **GitHub Markdown preview** → [../UML_CLASS_DIAGRAMS_2026-05-13.md](../UML_CLASS_DIAGRAMS_2026-05-13.md) / [../ARCHITECTURE_FLOWS.md](../ARCHITECTURE_FLOWS.md) (Mermaid).
- **Engineering UML reference** → [../diagrams/*.png](../diagrams/) (matplotlib UML style).
