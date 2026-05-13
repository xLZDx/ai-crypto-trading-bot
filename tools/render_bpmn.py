"""
BPMN 2.0 XML generator — business-process diagrams for the trading bot.

Outputs renderable BPMN 2.0 files (bpmn-js / Camunda Modeler / Draw.io)
covering the three highest-decision-density flows:

  - core/bpmn/training_flow.bpmn  — operator → KPI gate → trainer → retire decision
  - core/bpmn/trading_flow.bpmn   — WS tick → 9 risk gates → order
  - core/bpmn/registry_claim.bpmn — claim_role atomic decision tree

Why BPMN over Mermaid:
  - First-class **gateways** (XOR/AND) with explicit decision semantics
  - First-class **swimlanes** (zones of responsibility)
  - First-class **parallel flows** (parallel gateway)
  - Industry-standard XML format — every BPMN tool reads it

Run:
    python tools/render_bpmn.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

OUT_DIR = Path(__file__).resolve().parents[1] / 'core' / 'bpmn'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Layout constants ───────────────────────────────────────────────────────
LANE_LABEL_W = 30        # vertical text label strip on left
LANE_BODY_W  = 1700      # main lane content width
LANE_W       = LANE_LABEL_W + LANE_BODY_W
LANE_H       = 130
NODE_GAP_X   = 50        # horizontal gap between nodes
NODE_PAD_X   = 30        # left padding inside a lane

# Sizes per BPMN spec
EVT_SIZE   = 36          # start / end event (circle)
TASK_W     = 110
TASK_H     = 80
GW_SIZE    = 50          # gateway diamond


NodeKind = Literal['startEvent', 'endEvent', 'task', 'userTask', 'serviceTask',
                   'exclusiveGateway', 'parallelGateway']


@dataclass
class Node:
    id: str
    kind: NodeKind
    name: str
    lane: str
    col: int                  # 0-indexed horizontal position within the lane
    # Computed during layout
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    label: str = ''
    condition: Optional[str] = None    # for XOR-gateway-out edges


@dataclass
class Lane:
    id: str
    name: str


@dataclass
class Flow:
    """One BPMN process."""
    id: str
    name: str
    lanes: list[Lane]
    nodes: list[Node]
    edges: list[Edge]


# ─── XML helpers ────────────────────────────────────────────────────────────
def _xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;'))


def _node_size(kind: NodeKind) -> tuple[float, float]:
    if kind in ('startEvent', 'endEvent'):
        return EVT_SIZE, EVT_SIZE
    if kind in ('exclusiveGateway', 'parallelGateway'):
        return GW_SIZE, GW_SIZE
    return TASK_W, TASK_H


def _layout(flow: Flow) -> None:
    """Assign x/y/w/h to every node based on its (lane, col)."""
    lane_index = {lane.id: i for i, lane in enumerate(flow.lanes)}
    # Build a per-lane sorted column list so column positions tile cleanly.
    by_lane: dict[str, list[Node]] = {lane.id: [] for lane in flow.lanes}
    for n in flow.nodes:
        if n.lane not in by_lane:
            raise ValueError(f"Node {n.id!r} references unknown lane {n.lane!r}")
        by_lane[n.lane].append(n)
    for lane in flow.lanes:
        lst = sorted(by_lane[lane.id], key=lambda n: n.col)
        cursor_x = LANE_LABEL_W + NODE_PAD_X
        last_col = -1
        for n in lst:
            w, h = _node_size(n.kind)
            # Bump cursor for columns the previous node skipped.
            if n.col > last_col + 1:
                cursor_x += (n.col - last_col - 1) * (TASK_W + NODE_GAP_X)
            n.w, n.h = w, h
            # Center inside the lane vertically (use TASK_H as visual band height)
            li = lane_index[lane.id]
            lane_top = li * LANE_H
            n.y = lane_top + (LANE_H - h) / 2
            n.x = cursor_x
            cursor_x += w + NODE_GAP_X
            last_col = n.col


def _edge_waypoints(src: Node, dst: Node) -> list[tuple[float, float]]:
    """Compute orthogonal routing for one edge.

    Strategy:
      - If src and dst are in the same horizontal band (same lane → same y),
        route as a single horizontal segment with a 1-bend dogleg if needed.
      - If they're in different lanes, route src.right → midpoint → dst.left
        with a vertical segment at the midpoint x.
    """
    sx = src.x + src.w        # right edge of src
    sy = src.y + src.h / 2
    tx = dst.x                # left edge of dst
    ty = dst.y + dst.h / 2

    # Going BACKWARDS (loop) — route under or over
    if tx < sx - 5:
        # Loop back via a horizontal lower band
        mid_y = max(sy, ty) + 60
        return [
            (sx, sy), (sx + 25, sy), (sx + 25, mid_y),
            (tx - 25, mid_y), (tx - 25, ty), (tx, ty),
        ]
    if abs(sy - ty) < 5:
        # Same horizontal band → simple two-point line
        return [(sx, sy), (tx, ty)]
    # Different lanes → orthogonal dogleg through a mid-x
    mid_x = (sx + tx) / 2
    return [(sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty)]


def _shape_xml(n: Node) -> str:
    label_extra = ''
    if n.kind in ('startEvent', 'endEvent', 'exclusiveGateway', 'parallelGateway'):
        # Add a label below the small shape so the name shows up.
        lx = n.x - 20
        ly = n.y + n.h + 4
        label_extra = (
            f'      <bpmndi:BPMNLabel>\n'
            f'        <dc:Bounds x="{lx:.0f}" y="{ly:.0f}" width="{TASK_W:.0f}" height="22"/>\n'
            f'      </bpmndi:BPMNLabel>\n'
        )
    return (
        f'    <bpmndi:BPMNShape id="Shape_{n.id}" bpmnElement="{n.id}">\n'
        f'      <dc:Bounds x="{n.x:.0f}" y="{n.y:.0f}" width="{n.w:.0f}" height="{n.h:.0f}"/>\n'
        f'{label_extra}'
        f'    </bpmndi:BPMNShape>\n'
    )


def _edge_xml(edge: Edge, nodes_by_id: dict[str, Node]) -> str:
    src = nodes_by_id[edge.src]
    dst = nodes_by_id[edge.dst]
    waypoints = _edge_waypoints(src, dst)
    points = ''.join(
        f'      <di:waypoint x="{x:.0f}" y="{y:.0f}"/>\n'
        for x, y in waypoints
    )
    label_xml = ''
    if edge.label:
        mid = waypoints[len(waypoints) // 2]
        lx, ly = mid[0] - 30, mid[1] - 18
        label_xml = (
            f'      <bpmndi:BPMNLabel>\n'
            f'        <dc:Bounds x="{lx:.0f}" y="{ly:.0f}" width="80" height="14"/>\n'
            f'      </bpmndi:BPMNLabel>\n'
        )
    return (
        f'    <bpmndi:BPMNEdge id="Edge_{edge.id}" bpmnElement="{edge.id}">\n'
        f'{points}'
        f'{label_xml}'
        f'    </bpmndi:BPMNEdge>\n'
    )


def _lane_xml(lane: Lane, i: int, total_w: float) -> str:
    """Lane visual shape — full-width horizontal band."""
    y = i * LANE_H
    return (
        f'    <bpmndi:BPMNShape id="Shape_{lane.id}" bpmnElement="{lane.id}" isHorizontal="true">\n'
        f'      <dc:Bounds x="0" y="{y:.0f}" width="{total_w:.0f}" height="{LANE_H:.0f}"/>\n'
        f'    </bpmndi:BPMNShape>\n'
    )


def _build_bpmn_xml(flow: Flow) -> str:
    _layout(flow)
    nodes_by_id = {n.id: n for n in flow.nodes}

    # Process body — flow nodes + sequence flows
    flow_node_xml = []
    for n in flow.nodes:
        # Pre-compute incoming/outgoing edge IDs for each node
        incoming = [e.id for e in flow.edges if e.dst == n.id]
        outgoing = [e.id for e in flow.edges if e.src == n.id]
        in_xml  = ''.join(f'      <bpmn:incoming>{eid}</bpmn:incoming>\n' for eid in incoming)
        out_xml = ''.join(f'      <bpmn:outgoing>{eid}</bpmn:outgoing>\n' for eid in outgoing)
        flow_node_xml.append(
            f'    <bpmn:{n.kind} id="{n.id}" name="{_xml_escape(n.name)}">\n'
            f'{in_xml}{out_xml}'
            f'    </bpmn:{n.kind}>\n'
        )
    flow_node_block = ''.join(flow_node_xml)

    seq_flow_xml = []
    for e in flow.edges:
        cond_xml = ''
        if e.condition:
            cond_xml = (
                f'      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">'
                f'{_xml_escape(e.condition)}</bpmn:conditionExpression>\n'
            )
        seq_flow_xml.append(
            f'    <bpmn:sequenceFlow id="{e.id}" sourceRef="{e.src}" targetRef="{e.dst}"'
            f' name="{_xml_escape(e.label)}">\n'
            f'{cond_xml}'
            f'    </bpmn:sequenceFlow>\n'
        )
    seq_flow_block = ''.join(seq_flow_xml)

    # Lane set — which flow nodes belong to which lane
    lane_xml = []
    for lane in flow.lanes:
        refs = ''.join(
            f'        <bpmn:flowNodeRef>{n.id}</bpmn:flowNodeRef>\n'
            for n in flow.nodes if n.lane == lane.id
        )
        lane_xml.append(
            f'      <bpmn:lane id="{lane.id}" name="{_xml_escape(lane.name)}">\n'
            f'{refs}'
            f'      </bpmn:lane>\n'
        )
    lane_block = ''.join(lane_xml)

    # Diagram interchange — shapes + edges
    total_w = LANE_W
    total_h = LANE_H * len(flow.lanes)
    di_shapes = ''.join(_lane_xml(lane, i, total_w) for i, lane in enumerate(flow.lanes))
    di_shapes += ''.join(_shape_xml(n) for n in flow.nodes)
    di_edges  = ''.join(_edge_xml(e, nodes_by_id) for e in flow.edges)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
                  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  id="Definitions_{flow.id}"
                  targetNamespace="http://aitrader.local/bpmn">
  <bpmn:process id="{flow.id}" name="{_xml_escape(flow.name)}" isExecutable="false">
    <bpmn:laneSet id="LaneSet_{flow.id}">
{lane_block}    </bpmn:laneSet>
{flow_node_block}{seq_flow_block}  </bpmn:process>
  <bpmndi:BPMNDiagram id="Diagram_{flow.id}">
    <bpmndi:BPMNPlane id="Plane_{flow.id}" bpmnElement="{flow.id}">
{di_shapes}{di_edges}    </bpmndi:BPMNPlane>
  </bpmndi:BPMNDiagram>
</bpmn:definitions>
'''


def _save(flow: Flow) -> Path:
    path = OUT_DIR / f'{flow.id}.bpmn'
    path.write_text(_build_bpmn_xml(flow), encoding='utf-8')
    print(f"  [OK] {path}")
    return path


# ─── Flow 1: Training pipeline ──────────────────────────────────────────────
def training_flow() -> Flow:
    lanes = [
        Lane('Lane_Operator',      'Operator'),
        Lane('Lane_Dashboard',     'Dashboard + Orchestrator'),
        Lane('Lane_KPIGate_Pre',   'KPI Gate (pre-flight)'),
        Lane('Lane_MLE_Pre',       'ML Engineer Agent (pre)'),
        Lane('Lane_Trainer',       'Trainer Agent (lane worker)'),
        Lane('Lane_MLE_Post',      'ML Engineer Agent (post)'),
        Lane('Lane_KPIGate_Post',  'KPI Gate (3-strike)'),
    ]
    n = [
        # Operator row
        Node('Start_Click', 'startEvent', 'Click Train all', 'Lane_Operator', col=0),
        Node('End_Reject',  'endEvent',   'Rejected',         'Lane_Operator', col=6),
        Node('End_Keep',    'endEvent',   'KEEP',             'Lane_Operator', col=12),
        Node('End_Review',  'endEvent',   'REVIEW',           'Lane_Operator', col=13),
        Node('End_Retire',  'endEvent',   'RETIRE',           'Lane_Operator', col=14),

        # Dashboard row
        Node('Task_Submit',  'serviceTask', 'POST /api/training/run\nper (model, tf)',
             'Lane_Dashboard', col=1),

        # KPI gate pre
        Node('GW_IsRetired', 'exclusiveGateway', 'is_retired?', 'Lane_KPIGate_Pre', col=2),

        # MLE pre  — enriched with production threshold values
        Node('Task_Validators', 'serviceTask',
             'Run 5 validators:\n• freshness ≤ TF×3\n• label_imbalance ≥ 5%\n• nan_density < 5%\n• drift z-score < 2.0\n• feature_count = 23',
             'Lane_MLE_Pre', col=3),
        Node('GW_AllValPass', 'exclusiveGateway', 'all pass?', 'Lane_MLE_Pre', col=4),

        # Trainer — enriched with Triple Barrier constants
        Node('Task_TrainPipe', 'serviceTask',
             '9-step pipeline:\n• Triple Barrier pt=2.5 sl=1.5\n  max_bars=12\n• 60/20/20 split + 12-bar purge\n• PurgedKFold 5-fold\n  embargo = 2×max_bars / N\n• CalibratedClassifierCV isotonic\n• Sortino search [0.40-0.70]\n• HMAC-SHA256 sign\n• write <key>_meta.json',
             'Lane_Trainer', col=7),

        # MLE post
        Node('Task_PSR', 'serviceTask',
             'Bailey-LdP PSR:\nz = (SR - bench)√(n-1)\n  / √(1 - skew·SR\n     + (k-1)/4·SR²)\nPSR < 0.95 → REVIEW',
             'Lane_MLE_Post', col=8),

        # KPI gate post  — explicit thresholds from training_rules.json:kpi_threshold
        Node('Task_AppendRun', 'serviceTask',
             'Append run row to\ntraining_runs/\n<model>__<tf>.parquet',
             'Lane_KPIGate_Post', col=9),
        Node('GW_KPIPass', 'exclusiveGateway',
             'wf_acc ≥ 50%\nAND wf_win_rate ≥ 35%\nAND wf_total_trades ≥ 30?',
             'Lane_KPIGate_Post', col=10),
        Node('GW_3Strike', 'exclusiveGateway',
             'last 3 runs all\nbelow threshold?',
             'Lane_KPIGate_Post', col=11),
        Node('Task_Retire', 'serviceTask',
             'Add to\nretired_models.json',
             'Lane_KPIGate_Post', col=12),
    ]
    e = [
        Edge('f1',  'Start_Click',     'Task_Submit'),
        Edge('f2',  'Task_Submit',     'GW_IsRetired'),
        Edge('f3a', 'GW_IsRetired',    'End_Reject',     label='retired', condition='retired=true'),
        Edge('f3b', 'GW_IsRetired',    'Task_Validators', label='active'),
        Edge('f4',  'Task_Validators', 'GW_AllValPass'),
        Edge('f5a', 'GW_AllValPass',   'End_Reject',     label='any fail'),
        Edge('f5b', 'GW_AllValPass',   'Task_TrainPipe', label='all pass'),
        Edge('f6',  'Task_TrainPipe',  'Task_PSR'),
        Edge('f7',  'Task_PSR',        'Task_AppendRun'),
        Edge('f8',  'Task_AppendRun',  'GW_KPIPass'),
        Edge('f9a', 'GW_KPIPass',      'End_Keep',       label='all pass'),
        Edge('f9b', 'GW_KPIPass',      'GW_3Strike',     label='any miss'),
        Edge('f10a','GW_3Strike',      'End_Review',     label='no'),
        Edge('f10b','GW_3Strike',      'Task_Retire',    label='yes'),
        Edge('f11', 'Task_Retire',     'End_Retire'),
    ]
    return Flow('TrainingFlow',
                'Training pipeline — operator click to KEEP/REVIEW/RETIRE',
                lanes, n, e)


# ─── Flow 2: Live trading ───────────────────────────────────────────────────
def trading_flow() -> Flow:
    lanes = [
        Lane('Lane_Market',  'Market data (Binance WS)'),
        Lane('Lane_Signal',  'Signal generation'),
        Lane('Lane_Spec',    'Market specialists (parallel)'),
        Lane('Lane_Risk',    'Risk Agent (9-gate stack)'),
        Lane('Lane_Exec',    'Execution + Kill Switch'),
        Lane('Lane_Outcome', 'Order outcome'),
    ]
    n = [
        # Market
        Node('Start_Tick', 'startEvent', 'WS tick',  'Lane_Market', col=0),
        Node('Task_Update','serviceTask','MarketAnalyzer\nupdate state\n+ regime',
             'Lane_Market', col=1),

        # Signal
        Node('Task_RawSig','serviceTask','SignalAgent\ncompute raw signal',
             'Lane_Signal', col=2),
        Node('GW_Meta',    'exclusiveGateway', 'meta_pass?', 'Lane_Signal', col=3),
        Node('End_MetaBlock','endEvent','blocked by\nmeta-labeler', 'Lane_Signal', col=4),

        # Specialists
        Node('GW_FanOut',  'parallelGateway', 'split', 'Lane_Spec', col=5),
        Node('Task_Spot',  'serviceTask', 'SpotAgent\nconf >= 0.62', 'Lane_Spec', col=6),
        Node('Task_Fut',   'serviceTask', 'FuturesAgent\nfunding + liq gates', 'Lane_Spec', col=7),
        Node('Task_Scalp', 'serviceTask', 'ScalpingAgent\nROUND_TRIP_FEE check', 'Lane_Spec', col=8),
        Node('GW_FanIn',   'parallelGateway', 'join', 'Lane_Spec', col=9),

        # Risk
        Node('Task_9Gate', 'serviceTask',
             '9-gate stack:\nfreshness → API latency →\ncircuit → drawdown → daily →\nliquidity → β → LLM → Kelly',
             'Lane_Risk', col=10),
        Node('GW_AllGates', 'exclusiveGateway', 'all gates pass?', 'Lane_Risk', col=11),
        Node('End_RiskBlock','endEvent','blocked', 'Lane_Risk', col=12),

        # Exec
        Node('GW_KS',      'exclusiveGateway', 'KillSwitch\npaused?', 'Lane_Exec', col=13),
        Node('Task_Place', 'serviceTask', 'ExecutionAgent\nplace order on\nBinance (CCXT)',
             'Lane_Exec', col=14),
        Node('End_KSBlock','endEvent','blocked by\nkill switch', 'Lane_Exec', col=15),

        # Outcome
        Node('Task_Fill',  'serviceTask', 'fill received\n+ P&L feedback\n+ DB persist',
             'Lane_Outcome', col=16),
        Node('End_Done',   'endEvent', 'cycle complete', 'Lane_Outcome', col=17),
    ]
    e = [
        Edge('t1',  'Start_Tick',    'Task_Update'),
        Edge('t2',  'Task_Update',   'Task_RawSig'),
        Edge('t3',  'Task_RawSig',   'GW_Meta'),
        Edge('t4a', 'GW_Meta',       'End_MetaBlock', label='BLOCK'),
        Edge('t4b', 'GW_Meta',       'GW_FanOut',     label='PASS'),
        Edge('t5a', 'GW_FanOut',     'Task_Spot'),
        Edge('t5b', 'GW_FanOut',     'Task_Fut'),
        Edge('t5c', 'GW_FanOut',     'Task_Scalp'),
        Edge('t6a', 'Task_Spot',     'GW_FanIn'),
        Edge('t6b', 'Task_Fut',      'GW_FanIn'),
        Edge('t6c', 'Task_Scalp',    'GW_FanIn'),
        Edge('t7',  'GW_FanIn',      'Task_9Gate'),
        Edge('t8',  'Task_9Gate',    'GW_AllGates'),
        Edge('t9a', 'GW_AllGates',   'End_RiskBlock', label='any fail'),
        Edge('t9b', 'GW_AllGates',   'GW_KS',         label='all pass'),
        Edge('t10a','GW_KS',         'End_KSBlock',   label='paused'),
        Edge('t10b','GW_KS',         'Task_Place',    label='running'),
        Edge('t11', 'Task_Place',    'Task_Fill'),
        Edge('t12', 'Task_Fill',     'End_Done'),
    ]
    return Flow('TradingFlow',
                'Live trading — one WS tick to filled order',
                lanes, n, e)


# ─── Flow 3: Process registry claim_role decision tree ─────────────────────
def registry_claim_flow() -> Flow:
    lanes = [
        Lane('Lane_Caller',      'Caller (bot / dashboard / cluster_orch)'),
        Lane('Lane_Lock',        'safe_json.transaction (FileLock)'),
        Lane('Lane_Checks',      'Liveness checks'),
        Lane('Lane_Decision',    'Claim decision'),
        Lane('Lane_Audit',       'Audit log + side-effects'),
    ]
    n = [
        Node('Start_Call', 'startEvent', 'claim_role(role)', 'Lane_Caller', col=0),
        Node('Task_Lock',  'serviceTask', 'FileLock\n(5s timeout)\n+ json.load',
             'Lane_Lock', col=1),
        Node('GW_Existing','exclusiveGateway', 'existing entry?', 'Lane_Checks', col=2),
        Node('GW_SamePid', 'exclusiveGateway', 'same PID\n(re-entrant)?', 'Lane_Checks', col=3),
        Node('Task_Alive', 'serviceTask', '_pid_alive(pid)\nvia psutil\n(zombie filter)',
             'Lane_Checks', col=4),
        Node('GW_AliveFresh','exclusiveGateway', 'alive AND\nfresh<300s?', 'Lane_Checks', col=5),
        Node('Task_Reap',  'serviceTask', 'audit reap\n(stale entry)',
             'Lane_Decision', col=6),
        Node('Task_WriteEntry','serviceTask', 'write {pid, cmdline,\nhost, heartbeat_ts}\n+ audit claim',
             'Lane_Decision', col=7),
        Node('Task_BlockedAudit','serviceTask', 'audit claim_blocked', 'Lane_Decision', col=8),
        Node('Task_Atomic','serviceTask', 'tempfile + os.replace\n→ release FileLock',
             'Lane_Lock', col=9),
        Node('Task_Sideeffects','serviceTask', 'logger.warning/info\n+ _append_audit_log\n(after lock release)',
             'Lane_Audit', col=10),
        Node('End_OK',     'endEvent', '(True, entry)', 'Lane_Caller', col=11),
        Node('End_Blocked','endEvent', '(False, existing)\n→ caller os._exit(0)', 'Lane_Caller', col=12),
    ]
    e = [
        Edge('r1',  'Start_Call',       'Task_Lock'),
        Edge('r2',  'Task_Lock',        'GW_Existing'),
        Edge('r3a', 'GW_Existing',      'Task_WriteEntry', label='no'),
        Edge('r3b', 'GW_Existing',      'GW_SamePid',      label='yes'),
        Edge('r4a', 'GW_SamePid',       'End_OK',          label='yes (no-op)'),
        Edge('r4b', 'GW_SamePid',       'Task_Alive',      label='no'),
        Edge('r5',  'Task_Alive',       'GW_AliveFresh'),
        Edge('r6a', 'GW_AliveFresh',    'Task_BlockedAudit', label='live + fresh'),
        Edge('r6b', 'GW_AliveFresh',    'Task_Reap',         label='stale'),
        Edge('r7',  'Task_Reap',        'Task_WriteEntry'),
        Edge('r8',  'Task_WriteEntry',  'Task_Atomic'),
        Edge('r9',  'Task_BlockedAudit','Task_Atomic'),
        Edge('r10', 'Task_Atomic',      'Task_Sideeffects'),
        Edge('r11a','Task_Sideeffects', 'End_OK',           label='claimed'),
        Edge('r11b','Task_Sideeffects', 'End_Blocked',      label='blocked'),
    ]
    return Flow('RegistryClaim',
                'process_registry.claim_role — atomic decision tree',
                lanes, n, e)


# ─── Flow 4: Models lifecycle — load + HMAC verify + predict ────────────────
def models_lifecycle_flow() -> Flow:
    lanes = [
        Lane('Lane_Agent',  'Caller agent (Spot / Futures / Signal)'),
        Lane('Lane_MP',     'MLPredictor'),
        Lane('Lane_FS',     'Filesystem (D:/models)'),
        Lane('Lane_MI',     'model_integrity (HMAC-SHA256)'),
        Lane('Lane_Model',  'Trained model artefact'),
        Lane('Lane_Result', 'Return path'),
    ]
    n = [
        Node('Start_Predict', 'startEvent', 'predict_proba(features)', 'Lane_Agent', col=0),
        Node('Task_OpenMeta', 'serviceTask', 'open\n<key>_meta.json', 'Lane_MP', col=1),
        Node('GW_FileExists', 'exclusiveGateway', 'file exists?', 'Lane_MP', col=2),
        Node('End_NoFile',    'endEvent', 'is_loaded=False\n(model missing)', 'Lane_Result', col=3),
        Node('Task_ReadMeta', 'serviceTask', 'parse meta:\nfeatures, threshold,\nsignature_hex', 'Lane_MP', col=3),
        Node('Task_OpenJoblib','serviceTask', 'open joblib\nbinary blob', 'Lane_FS', col=4),
        Node('Task_VerifyHMAC','serviceTask', 'verify_and_load_bytes\nHMAC-SHA256\nvs MODEL_SIGNING_KEY', 'Lane_MI', col=5),
        Node('GW_SigValid',   'exclusiveGateway', 'signature valid?\n(compare_digest)', 'Lane_MI', col=6),
        Node('End_SigFail',   'endEvent', 'SignatureError\nREFUSE to load', 'Lane_Result', col=7),
        Node('Task_JoblibLoad','serviceTask', 'joblib.load\n(BytesIO bytes)', 'Lane_MP', col=7),
        Node('Task_BuildX',   'serviceTask', 'build DataFrame\n[meta.features]\nNaN-fill 0', 'Lane_MP', col=8),
        Node('Task_Predict',  'serviceTask', 'model.predict_proba(X)', 'Lane_Model', col=9),
        Node('End_Return',    'endEvent', 'return float p_win', 'Lane_Result', col=10),
    ]
    e = [
        Edge('m1',  'Start_Predict',   'Task_OpenMeta'),
        Edge('m2',  'Task_OpenMeta',   'GW_FileExists'),
        Edge('m3a', 'GW_FileExists',   'End_NoFile',     label='no'),
        Edge('m3b', 'GW_FileExists',   'Task_ReadMeta',  label='yes'),
        Edge('m4',  'Task_ReadMeta',   'Task_OpenJoblib'),
        Edge('m5',  'Task_OpenJoblib', 'Task_VerifyHMAC'),
        Edge('m6',  'Task_VerifyHMAC', 'GW_SigValid'),
        Edge('m7a', 'GW_SigValid',     'End_SigFail',    label='mismatch'),
        Edge('m7b', 'GW_SigValid',     'Task_JoblibLoad',label='valid'),
        Edge('m8',  'Task_JoblibLoad', 'Task_BuildX'),
        Edge('m9',  'Task_BuildX',     'Task_Predict'),
        Edge('m10', 'Task_Predict',    'End_Return'),
    ]
    return Flow('ModelsLifecycle',
                'Models lifecycle — MLPredictor: load → HMAC verify → predict_proba',
                lanes, n, e)


# ─── Flow 5: Infrastructure startup — restart_all.ps1 ───────────────────────
def infra_startup_flow() -> Flow:
    lanes = [
        Lane('Lane_Op',      'Operator'),
        Lane('Lane_Script',  'restart_all.ps1'),
        Lane('Lane_Preflight','Pre-flight (early-kill + recovery)'),
        Lane('Lane_Tier1',   'Tier-1 processes (operational)'),
        Lane('Lane_Tier2',   'Tier-2 processes (ancillary)'),
        Lane('Lane_Done',    'System ready'),
    ]
    n = [
        Node('Start_Boot', 'startEvent', 'operator runs\nrestart_all.ps1',
             'Lane_Op', col=0),
        Node('Task_EarlyKill', 'serviceTask',
             'Early-kill pre-step:\nStop-Process bot/dash/\norderbook_collector/...',
             'Lane_Preflight', col=1),
        Node('Task_Parquet', 'serviceTask',
             'Parquet store check\n(DuckDB import + dir)',
             'Lane_Preflight', col=2),
        Node('Task_Recovery', 'serviceTask',
             'startup_recovery\n--archive-only\n(5-min cap)',
             'Lane_Preflight', col=3),
        Node('GW_FanOut1', 'parallelGateway', 'spawn Tier-1', 'Lane_Script', col=4),
        Node('Task_Monitor',  'serviceTask', 'monitor :5001',     'Lane_Tier1', col=5),
        Node('Task_Cluster',  'serviceTask', 'cluster_orch :7700','Lane_Tier1', col=6),
        Node('Task_Dash',     'serviceTask', 'dashboard :5000',   'Lane_Tier1', col=7),
        Node('Task_Bot',      'serviceTask', 'bot (src.main)',    'Lane_Tier1', col=8),
        Node('Task_RT',       'serviceTask', 'realtime_db_writer','Lane_Tier1', col=9),
        Node('Task_OB',       'serviceTask', 'orderbook_collector','Lane_Tier1', col=10),
        Node('GW_FanIn1', 'parallelGateway', 'tier-1 ready', 'Lane_Script', col=11),
        Node('GW_FanOut2', 'parallelGateway', 'spawn Tier-2', 'Lane_Script', col=12),
        Node('Task_OBW',      'serviceTask', 'orderbook_writer\n(X1.2)','Lane_Tier2', col=13),
        Node('Task_WLD',      'serviceTask', 'watchlist_downloader','Lane_Tier2', col=14),
        Node('Task_DOR',      'serviceTask', 'data_orchestrator', 'Lane_Tier2', col=15),
        Node('Task_DEB',      'serviceTask', 'debug_supervisor',  'Lane_Tier2', col=13),
        Node('Task_DW',       'serviceTask', 'dashboard_watchdog','Lane_Tier2', col=14),
        Node('Task_SW',       'serviceTask', 'sweep_watchdog',    'Lane_Tier2', col=15),
        Node('GW_FanIn2', 'parallelGateway', 'all 11 spawned', 'Lane_Script', col=16),
        Node('End_Ready', 'endEvent', '11 roles claimed\nin process_registry',
             'Lane_Done', col=17),
    ]
    # Parallel fan-out edges (Tier 1)
    e = [
        Edge('s1', 'Start_Boot', 'Task_EarlyKill'),
        Edge('s2', 'Task_EarlyKill', 'Task_Parquet'),
        Edge('s3', 'Task_Parquet', 'Task_Recovery'),
        Edge('s4', 'Task_Recovery', 'GW_FanOut1'),
        Edge('s5a','GW_FanOut1', 'Task_Monitor'),
        Edge('s5b','GW_FanOut1', 'Task_Cluster'),
        Edge('s5c','GW_FanOut1', 'Task_Dash'),
        Edge('s5d','GW_FanOut1', 'Task_Bot'),
        Edge('s5e','GW_FanOut1', 'Task_RT'),
        Edge('s5f','GW_FanOut1', 'Task_OB'),
        Edge('s6a','Task_Monitor','GW_FanIn1'),
        Edge('s6b','Task_Cluster','GW_FanIn1'),
        Edge('s6c','Task_Dash',   'GW_FanIn1'),
        Edge('s6d','Task_Bot',    'GW_FanIn1'),
        Edge('s6e','Task_RT',     'GW_FanIn1'),
        Edge('s6f','Task_OB',     'GW_FanIn1'),
        Edge('s7', 'GW_FanIn1', 'GW_FanOut2'),
        # Tier-2 fan-out — second pass after Tier-1 health
        Edge('s8a','GW_FanOut2', 'Task_OBW'),
        Edge('s8b','GW_FanOut2', 'Task_WLD'),
        Edge('s8c','GW_FanOut2', 'Task_DOR'),
        Edge('s8d','GW_FanOut2', 'Task_DEB'),
        Edge('s8e','GW_FanOut2', 'Task_DW'),
        Edge('s8f','GW_FanOut2', 'Task_SW'),
        Edge('s9a','Task_OBW','GW_FanIn2'),
        Edge('s9b','Task_WLD','GW_FanIn2'),
        Edge('s9c','Task_DOR','GW_FanIn2'),
        Edge('s9d','Task_DEB','GW_FanIn2'),
        Edge('s9e','Task_DW', 'GW_FanIn2'),
        Edge('s9f','Task_SW', 'GW_FanIn2'),
        Edge('s10','GW_FanIn2', 'End_Ready'),
    ]
    return Flow('InfraStartup',
                'Infrastructure startup — restart_all.ps1 to 11 roles claimed',
                lanes, n, e)


# ─── Flow 6: Agent lifecycle — BaseAgent thread loop ────────────────────────
def agent_lifecycle_flow() -> Flow:
    lanes = [
        Lane('Lane_Main',    'Caller (src/main.py)'),
        Lane('Lane_Init',    'BaseAgent.__init__'),
        Lane('Lane_Subs',    'AgentBus subscriptions'),
        Lane('Lane_Thread',  'Background thread (daemon)'),
        Lane('Lane_Status',  'agent_status.json + heartbeat'),
        Lane('Lane_End',     'Shutdown'),
    ]
    n = [
        Node('Start_New', 'startEvent', 'SpotAgent(...)\nor similar',
             'Lane_Main', col=0),
        Node('Task_Init', 'serviceTask',
             'self.bus, interval_sec,\n_running=False',
             'Lane_Init', col=1),
        Node('Task_Subs', 'serviceTask',
             'bus.subscribe("signal", _on_signal)\nbus.subscribe("regime", _on_regime)',
             'Lane_Subs', col=2),
        Node('Task_Start', 'serviceTask', 'agent.start()\n→ _running=True',
             'Lane_Main', col=3),
        Node('Task_Spawn', 'serviceTask', 'Thread(target=_loop,\ndaemon=True).start()',
             'Lane_Thread', col=4),
        Node('Task_WriteRunning', 'serviceTask',
             '_write_agent_status\n("running", interval_sec)\n(holds _status_write_lock)',
             'Lane_Status', col=5),
        Node('Task_RunCycle', 'serviceTask',
             '_run_cycle()\n(subclass-specific)',
             'Lane_Thread', col=6),
        Node('GW_CycleEx', 'exclusiveGateway', 'exception?', 'Lane_Thread', col=7),
        Node('Task_LogErr', 'serviceTask',
             'logger.error\n_write_agent_status\n("error")',
             'Lane_Status', col=8),
        Node('Task_WriteIdle', 'serviceTask',
             '_write_agent_status\n("idle")',
             'Lane_Status', col=9),
        Node('Task_Sleep', 'serviceTask', 'time.sleep(interval_sec)',
             'Lane_Thread', col=10),
        Node('GW_StillRun', 'exclusiveGateway', 'self._running?', 'Lane_Thread', col=11),
        Node('Task_Stop', 'serviceTask', 'agent.stop()\n→ _running=False',
             'Lane_Main', col=12),
        Node('End_Exit', 'endEvent', 'thread exits cleanly\n(daemon)',
             'Lane_End', col=13),
    ]
    e = [
        Edge('a1', 'Start_New',         'Task_Init'),
        Edge('a2', 'Task_Init',         'Task_Subs'),
        Edge('a3', 'Task_Subs',         'Task_Start'),
        Edge('a4', 'Task_Start',        'Task_Spawn'),
        Edge('a5', 'Task_Spawn',        'Task_WriteRunning'),
        Edge('a6', 'Task_WriteRunning', 'Task_RunCycle'),
        Edge('a7', 'Task_RunCycle',     'GW_CycleEx'),
        Edge('a8a','GW_CycleEx',        'Task_LogErr',     label='yes'),
        Edge('a8b','GW_CycleEx',        'Task_WriteIdle',  label='no'),
        Edge('a9', 'Task_LogErr',       'Task_Sleep'),
        Edge('a10','Task_WriteIdle',    'Task_Sleep'),
        Edge('a11','Task_Sleep',        'GW_StillRun'),
        Edge('a12a','GW_StillRun',      'Task_WriteRunning', label='yes (loop)'),
        Edge('a12b','GW_StillRun',      'Task_Stop',         label='no (stop called)'),
        Edge('a13','Task_Stop',         'End_Exit'),
    ]
    return Flow('AgentLifecycle',
                'BaseAgent lifecycle — init → loop → stop',
                lanes, n, e)


# ─── Flow 7: Trainer dispatch (factory + class hierarchy) ──────────────────
def trainer_dispatch_flow() -> Flow:
    lanes = [
        Lane('Lane_CO',     'cluster_orch :7700'),
        Lane('Lane_Factory','get_trainer_agent factory'),
        Lane('Lane_Reg',    'TRAINER_AGENT_REGISTRY'),
        Lane('Lane_TA',     'TrainerXAgent (concrete)'),
        Lane('Lane_Pipe',   'Train function pipeline'),
        Lane('Lane_CIO',    'cio_overrides.merge_with_defaults'),
        Lane('Lane_Sign',   'model_integrity + meta JSON'),
        Lane('Lane_Result', 'Result'),
    ]
    n = [
        Node('Start_Dispatch', 'startEvent', 'cluster_orch\ndispatch (model_key, tf)',
             'Lane_CO', col=0),
        Node('Task_Lookup', 'serviceTask', 'get_trainer_agent(key)',
             'Lane_Factory', col=1),
        Node('Task_RegRead', 'serviceTask', 'REGISTRY[key]', 'Lane_Reg', col=2),
        Node('GW_KnownKey', 'exclusiveGateway', 'known key?', 'Lane_Reg', col=3),
        Node('End_KeyErr', 'endEvent', 'KeyError\n"No trainer agent for ..."',
             'Lane_Result', col=4),
        Node('Task_Instantiate', 'serviceTask', 'instantiate fresh\n(NOT singleton)',
             'Lane_Factory', col=5),
        Node('Task_TrainCall', 'serviceTask', 'train(rules_version,\nn_samples_min)',
             'Lane_TA', col=6),
        Node('Task_ReadRules', 'serviceTask', 'read params +\ncio_overrides\nfrom training_rules.json',
             'Lane_Pipe', col=7),
        Node('Task_Merge', 'serviceTask',
             'merge_with_defaults\nschema-bounded:\n- drop wrong-type\n- drop out-of-range\n- drop non-allowlist',
             'Lane_CIO', col=8),
        Node('Task_RunPipe', 'serviceTask',
             'subclass train function\n(9-step pipeline)',
             'Lane_Pipe', col=9),
        Node('GW_Raised', 'exclusiveGateway', 'train raised?', 'Lane_Pipe', col=10),
        Node('Task_LastResultFail', 'serviceTask',
             'last_result =\n{ok: False, error: str(e)}',
             'Lane_TA', col=11),
        Node('End_Fail', 'endEvent', '(False, {error})',
             'Lane_Result', col=12),
        Node('Task_Sign', 'serviceTask', 'sign_model(joblib_path)\nHMAC-SHA256',
             'Lane_Sign', col=11),
        Node('Task_WriteMeta', 'serviceTask',
             'write <key>_meta.json\n{wf_acc, threshold,\ncio_overrides_applied,\nsignature_hex}',
             'Lane_Sign', col=12),
        Node('Task_UpdateTask', 'serviceTask',
             'cluster_orch.\nupdate_task("done",\nmeta_path)',
             'Lane_CO', col=13),
        Node('End_OK', 'endEvent', '(True, info, meta_path)',
             'Lane_Result', col=14),
    ]
    e = [
        Edge('d1', 'Start_Dispatch',  'Task_Lookup'),
        Edge('d2', 'Task_Lookup',     'Task_RegRead'),
        Edge('d3', 'Task_RegRead',    'GW_KnownKey'),
        Edge('d4a','GW_KnownKey',     'End_KeyErr',         label='no'),
        Edge('d4b','GW_KnownKey',     'Task_Instantiate',   label='yes'),
        Edge('d5', 'Task_Instantiate','Task_TrainCall'),
        Edge('d6', 'Task_TrainCall',  'Task_ReadRules'),
        Edge('d7', 'Task_ReadRules',  'Task_Merge'),
        Edge('d8', 'Task_Merge',      'Task_RunPipe'),
        Edge('d9', 'Task_RunPipe',    'GW_Raised'),
        Edge('d10a','GW_Raised',      'Task_LastResultFail',label='yes'),
        Edge('d10b','GW_Raised',      'Task_Sign',          label='no'),
        Edge('d11', 'Task_LastResultFail','End_Fail'),
        Edge('d12', 'Task_Sign',      'Task_WriteMeta'),
        Edge('d13', 'Task_WriteMeta', 'Task_UpdateTask'),
        Edge('d14', 'Task_UpdateTask','End_OK'),
    ]
    return Flow('TrainerDispatch',
                'Trainer dispatch — factory → train → sign → meta JSON',
                lanes, n, e)


# ─── Flow 8: Risk gates — RiskAgent traverses 9 gates in order ─────────────
def risk_gates_flow() -> Flow:
    lanes = [
        Lane('Lane_Bus',     'AgentBus (input)'),
        Lane('Lane_Pretrade','Pre-trade gates (1-3)'),
        Lane('Lane_DDgates', 'Drawdown gates (4-5)'),
        Lane('Lane_Liquid',  'Liquidity + β gates (6-7)'),
        Lane('Lane_LLM',     'AgenticLLM macro veto (8)'),
        Lane('Lane_Kelly',   'Kelly sizing (9)'),
        Lane('Lane_Order',   'Order publish'),
        Lane('Lane_Block',   'Block / kill paths'),
    ]
    n = [
        Node('Start_Sig', 'startEvent', 'receive\n"trade_signal"',
             'Lane_Bus', col=0),
        Node('GW_MetaDir', 'exclusiveGateway',
             'meta_pass=True AND\ndirection != 0?', 'Lane_Bus', col=1),
        Node('End_DropMeta', 'endEvent', 'drop (no log)', 'Lane_Block', col=2),

        Node('GW_G1', 'exclusiveGateway',
             'G1: data_freshness\nlast bar ≤ 300s?\n(DATA_STALE_SEC)', 'Lane_Pretrade', col=2),
        Node('GW_G2', 'exclusiveGateway',
             'G2: API latency\np99 < 500ms?\n(API_LATENCY_LIMIT_MS)', 'Lane_Pretrade', col=3),
        Node('GW_G3', 'exclusiveGateway',
             'G3: circuit breaker\n< 3 consec losses?\n(MAX_CONSECUTIVE_LOSSES)', 'Lane_Pretrade', col=4),

        Node('GW_G4', 'exclusiveGateway',
             'G4: cum drawdown\n< 10%?\n(MAX_DRAWDOWN_PCT)', 'Lane_DDgates', col=5),
        Node('Task_HardKill', 'serviceTask',
             '_hard_kill()\npublish "risk_kill_switch"\nflatten_all\n(STICKY pause)',
             'Lane_Block', col=6),
        Node('GW_G5', 'exclusiveGateway',
             'G5: daily loss\n< 5%?\n(MAX_DAILY_LOSS_PCT)', 'Lane_DDgates', col=6),

        Node('GW_G6', 'exclusiveGateway',
             'G6: liq_proximity\n< 0.85?\n(LIQ_PROXIMITY_BLOCK)', 'Lane_Liquid', col=7),
        Node('GW_G7', 'exclusiveGateway',
             'G7: BetaFilter\nwould_breach\n|β|_max = 1.0?', 'Lane_Liquid', col=8),

        Node('Task_LLM', 'serviceTask',
             'AgenticLLM.evaluate_trade\n• 60s TTL decision cache\n• threading.Lock + LRU 500\n• 11-model fallback chain',
             'Lane_LLM', col=9),
        Node('GW_G8', 'exclusiveGateway',
             'G8: LLM REJECTED?\n(fail-OPEN on quota:\nall cooled-down → APPROVED)',
             'Lane_LLM', col=10),

        Node('Task_Kelly', 'serviceTask',
             'G9: kelly.size\n• half_kelly = True\n• window = 50 trades\n• vol_scale × size_mult\n  (regime-conditional)',
             'Lane_Kelly', col=11),

        Node('Task_PubOrder', 'serviceTask',
             'publish "order"\n{pending, size_usdt}',
             'Lane_Order', col=12),
        Node('End_Order', 'endEvent', 'order dispatched to\nExecutionAgent',
             'Lane_Order', col=13),
        Node('End_Block', 'endEvent', 'BLOCKED', 'Lane_Block', col=8),
    ]
    e = [
        Edge('g1',  'Start_Sig',    'GW_MetaDir'),
        Edge('g2a', 'GW_MetaDir',   'End_DropMeta',   label='no'),
        Edge('g2b', 'GW_MetaDir',   'GW_G1',          label='yes'),
        Edge('g3a', 'GW_G1',        'End_Block',      label='stale'),
        Edge('g3b', 'GW_G1',        'GW_G2',          label='fresh'),
        Edge('g4a', 'GW_G2',        'End_Block',      label='slow'),
        Edge('g4b', 'GW_G2',        'GW_G3',          label='fast'),
        Edge('g5a', 'GW_G3',        'End_Block',      label='open'),
        Edge('g5b', 'GW_G3',        'GW_G4',          label='closed'),
        Edge('g6a', 'GW_G4',        'Task_HardKill',  label='breach'),
        Edge('g6b', 'GW_G4',        'GW_G5',          label='ok'),
        Edge('g7',  'Task_HardKill','End_Block'),
        Edge('g8a', 'GW_G5',        'Task_HardKill',  label='breach'),
        Edge('g8b', 'GW_G5',        'GW_G6',          label='ok'),
        Edge('g9a', 'GW_G6',        'End_Block',      label='close to liq'),
        Edge('g9b', 'GW_G6',        'GW_G7',          label='clear'),
        Edge('g10a','GW_G7',        'End_Block',      label='|β| breach'),
        Edge('g10b','GW_G7',        'Task_LLM',       label='ok'),
        Edge('g11', 'Task_LLM',     'GW_G8'),
        Edge('g12a','GW_G8',        'End_Block',      label='REJECTED'),
        Edge('g12b','GW_G8',        'Task_Kelly',     label='APPROVED'),
        Edge('g13', 'Task_Kelly',   'Task_PubOrder'),
        Edge('g14', 'Task_PubOrder','End_Order'),
    ]
    return Flow('RiskGates',
                'RiskAgent — 9-gate stack traversal (fail-closed defaults)',
                lanes, n, e)


def main() -> None:
    print('Rendering BPMN 2.0 XML files to', OUT_DIR)
    _save(training_flow())
    _save(trading_flow())
    _save(registry_claim_flow())
    _save(models_lifecycle_flow())
    _save(infra_startup_flow())
    _save(agent_lifecycle_flow())
    _save(trainer_dispatch_flow())
    _save(risk_gates_flow())
    print('Done.')


if __name__ == '__main__':
    main()
