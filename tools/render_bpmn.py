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

        # MLE pre
        Node('Task_Validators', 'serviceTask',
             'Run 5 validators:\nfreshness, label_imb,\nnan_density, drift z,\nfeature_count',
             'Lane_MLE_Pre', col=3),
        Node('GW_AllValPass', 'exclusiveGateway', 'all pass?', 'Lane_MLE_Pre', col=4),

        # Trainer
        Node('Task_TrainPipe', 'serviceTask',
             '9-step pipeline:\nTriple Barrier → 60/20/20\n→ PurgedKFold 5-fold\n→ Calibrated → Sortino\n→ HMAC sign → meta.json',
             'Lane_Trainer', col=7),

        # MLE post
        Node('Task_PSR', 'serviceTask',
             'PSR (Bailey-LdP)\n+ WF consistency\n+ baseline compare',
             'Lane_MLE_Post', col=8),

        # KPI gate post
        Node('Task_AppendRun', 'serviceTask',
             'Append run row to\ntraining_runs/\n<model>__<tf>.parquet',
             'Lane_KPIGate_Post', col=9),
        Node('GW_KPIPass', 'exclusiveGateway', 'all KPIs pass?', 'Lane_KPIGate_Post', col=10),
        Node('GW_3Strike', 'exclusiveGateway', '3 consec fails?', 'Lane_KPIGate_Post', col=11),
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


def main() -> None:
    print('Rendering BPMN 2.0 XML files to', OUT_DIR)
    _save(training_flow())
    _save(trading_flow())
    _save(registry_claim_flow())
    print('Done.')


if __name__ == '__main__':
    main()
