"""
Business-flow PNG renderer — presentation-quality images of every BPMN flow.

Operator (2026-05-13) needed slide-ready exports of the 8 business flows
(no browser / no Camunda Modeler / no Draw.io required). This renderer
reuses the Flow / Lane / Node / Edge dataclasses from render_bpmn.py
and draws each one as a BPMN-styled PNG using matplotlib.

Style rules (close to Draw.io's BPMN aesthetic):
  - Horizontal swimlanes with vertical text labels on the left
  - Tasks: rounded rectangles, light-blue fill (#dae8fc)
  - Start events: green-edge circles
  - End events: red-edge circles (thick border per BPMN spec)
  - XOR gateways: yellow diamonds with × inside
  - Parallel gateways: yellow diamonds with + inside
  - Sequence flows: orthogonal arrows with optional labels

Run:
    python tools/render_business_flow_pngs.py

Outputs to core/business_flows/*.png at 200 DPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / 'core' / 'business_flows'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import Flow definitions from render_bpmn.py (already shipped).
# Must register in sys.modules BEFORE exec_module — otherwise @dataclass
# inside render_bpmn.py can't resolve type annotations on Python 3.14
# (dataclasses._is_type does a sys.modules lookup on the owning module).
_TOOLS_DIR = str(PROJECT_ROOT / 'tools')
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import render_bpmn as rb  # noqa: E402

# ── Visual constants (Draw.io BPMN palette) ───────────────────────────────
TASK_FILL    = '#dae8fc'   # light blue
TASK_BORDER  = '#6c8ebf'   # medium blue
START_BORDER = '#82b366'   # green
END_BORDER   = '#b85450'   # red
GW_FILL      = '#fff2cc'   # pale yellow
GW_BORDER    = '#d79b00'   # amber
LANE_BG_A    = '#ffffff'
LANE_BG_B    = '#f5f5f5'   # alt-row striping
LANE_HDR_BG  = '#2b6cb0'   # dark blue header bar (lane label)
LANE_HDR_TXT = '#ffffff'
EDGE_COLOR   = '#444444'
LABEL_COLOR  = '#475569'
TITLE_COLOR  = '#0f172a'

# Layout: re-using rb's (lane, col) grid but with slightly larger metrics
# tuned for presentation slides.
LANE_LABEL_W = 56     # wider vertical text band so labels never overlap
TASK_W       = 130
TASK_H       = 70
EVT_SIZE     = 42
GW_SIZE      = 52
NODE_GAP_X   = 38
NODE_PAD_X   = 24
LANE_H       = 150    # taller lanes so rotated text fits


def _node_size(kind: str) -> tuple[float, float]:
    if kind in ('startEvent', 'endEvent'):
        return EVT_SIZE, EVT_SIZE
    if kind in ('exclusiveGateway', 'parallelGateway'):
        return GW_SIZE, GW_SIZE
    return TASK_W, TASK_H


def _layout(flow) -> None:
    """Assign x/y/w/h to every node based on (lane, col). Mirrors render_bpmn.

    Lane 0 is rendered at the TOP. (matplotlib y grows up, so we flip later
    by setting ax.invert_yaxis or computing y from total height.)"""
    lane_index = {lane.id: i for i, lane in enumerate(flow.lanes)}
    by_lane: dict[str, list] = {lane.id: [] for lane in flow.lanes}
    for n in flow.nodes:
        by_lane[n.lane].append(n)
    for lane in flow.lanes:
        lst = sorted(by_lane[lane.id], key=lambda n: n.col)
        cursor_x = LANE_LABEL_W + NODE_PAD_X
        last_col = -1
        for n in lst:
            w, h = _node_size(n.kind)
            if n.col > last_col + 1:
                cursor_x += (n.col - last_col - 1) * (TASK_W + NODE_GAP_X)
            n.w, n.h = w, h
            li = lane_index[n.lane]
            lane_top = li * LANE_H
            n.y = lane_top + (LANE_H - h) / 2
            n.x = cursor_x
            cursor_x += w + NODE_GAP_X
            last_col = n.col


def _edge_waypoints(src, dst) -> list[tuple[float, float]]:
    """Orthogonal routing — same logic as render_bpmn._edge_waypoints, but
    here y is flipped (matplotlib invert)."""
    sx = src.x + src.w
    sy = src.y + src.h / 2
    tx = dst.x
    ty = dst.y + dst.h / 2

    if tx < sx - 5:
        # Loop back via a lower band
        mid_y = max(sy, ty) + 70
        return [
            (sx, sy), (sx + 25, sy), (sx + 25, mid_y),
            (tx - 25, mid_y), (tx - 25, ty), (tx, ty),
        ]
    if abs(sy - ty) < 5:
        return [(sx, sy), (tx, ty)]
    mid_x = (sx + tx) / 2
    return [(sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty)]


def _wrap_lane_name(name: str, max_chars_per_line: int = 18) -> str:
    """Wrap a lane name onto 2 lines if longer than max_chars_per_line.
    Splits on '+' / '(' / space, preferring earlier breaks."""
    if len(name) <= max_chars_per_line:
        return name
    # Try splitting at '+' or '(' first (semantic break)
    for sep in (' + ', ' ('):
        if sep in name:
            a, b = name.split(sep, 1)
            return f'{a.strip()}\n{sep.strip()} {b.strip()}' if sep.strip() else f'{a.strip()}\n{b.strip()}'
    # Fallback: split at the space closest to the middle
    words = name.split()
    if len(words) >= 2:
        mid = len(words) // 2
        return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
    return name


def _draw_lane(ax, lane, lane_idx: int, total_lanes: int, total_w: float):
    """Horizontal swimlane band with a dark-blue vertical label strip on left."""
    y = lane_idx * LANE_H
    body_color = LANE_BG_A if lane_idx % 2 == 0 else LANE_BG_B
    ax.add_patch(Rectangle(
        (0, y), total_w, LANE_H,
        linewidth=1.0, edgecolor='#888', facecolor=body_color, zorder=1,
    ))
    ax.add_patch(Rectangle(
        (0, y), LANE_LABEL_W, LANE_H,
        linewidth=1.0, edgecolor='#444', facecolor=LANE_HDR_BG, zorder=2,
    ))
    # Wrap long names so they fit inside the LANE_H height after rotation.
    label = _wrap_lane_name(lane.name)
    ax.text(
        LANE_LABEL_W / 2, y + LANE_H / 2, label,
        ha='center', va='center', rotation=90,
        color=LANE_HDR_TXT, fontsize=8.5, fontweight='bold',
        multialignment='center', zorder=3,
    )


def _draw_task(ax, node):
    """Rounded rectangle, light blue, with multi-line task name."""
    body = FancyBboxPatch(
        (node.x, node.y), node.w, node.h,
        boxstyle="round,pad=0.02,rounding_size=8",
        linewidth=1.4, edgecolor=TASK_BORDER, facecolor=TASK_FILL, zorder=3,
    )
    ax.add_patch(body)
    ax.text(
        node.x + node.w / 2, node.y + node.h / 2, node.name,
        ha='center', va='center', fontsize=7.5, color=TITLE_COLOR,
        multialignment='center', zorder=4,
    )


def _draw_event(ax, node):
    """Start (green) / end (red) event circle."""
    cx, cy = node.x + node.w / 2, node.y + node.h / 2
    r = node.w / 2
    is_start = node.kind == 'startEvent'
    edge_col = START_BORDER if is_start else END_BORDER
    lw = 1.8 if is_start else 3.0
    ax.add_patch(Circle(
        (cx, cy), r, facecolor='white', edgecolor=edge_col, linewidth=lw, zorder=3,
    ))
    # Label BELOW the circle
    ax.text(
        cx, node.y + node.h + 4, node.name,
        ha='center', va='top', fontsize=7, color=LABEL_COLOR,
        multialignment='center', zorder=4,
    )


def _draw_gateway(ax, node):
    """Diamond gateway. × for XOR, + for parallel."""
    cx, cy = node.x + node.w / 2, node.y + node.h / 2
    s = node.w / 2
    diamond = Polygon(
        [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)],
        facecolor=GW_FILL, edgecolor=GW_BORDER, linewidth=1.6, zorder=3,
    )
    ax.add_patch(diamond)
    symbol = '×' if node.kind == 'exclusiveGateway' else '+'
    ax.text(
        cx, cy, symbol,
        ha='center', va='center', fontsize=18, fontweight='bold',
        color=GW_BORDER, zorder=4,
    )
    # Label ABOVE the diamond
    ax.text(
        cx, node.y - 4, node.name,
        ha='center', va='bottom', fontsize=7, color=LABEL_COLOR,
        multialignment='center', zorder=4,
    )


def _draw_edge(ax, edge, nodes_by_id: dict):
    src = nodes_by_id[edge.src]
    dst = nodes_by_id[edge.dst]
    wp = _edge_waypoints(src, dst)
    # Polyline segments (all but last)
    xs = [p[0] for p in wp]
    ys = [p[1] for p in wp]
    ax.add_line(Line2D(xs, ys, color=EDGE_COLOR, linewidth=1.2, zorder=2))
    # Arrowhead at the final segment
    ax.annotate(
        '', xy=wp[-1], xytext=wp[-2],
        arrowprops=dict(arrowstyle='-|>', color=EDGE_COLOR, lw=1.2,
                        mutation_scale=12),
        zorder=2,
    )
    # Edge label
    if edge.label:
        mid = wp[len(wp) // 2]
        # Drop label below the line for horizontal edges, beside it for verticals
        lx, ly = mid[0], mid[1] - 7
        ax.text(
            lx, ly, edge.label,
            fontsize=6.5, color=LABEL_COLOR, ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5),
            zorder=5,
        )


def render_flow(flow, fname: str | None = None) -> Path:
    """Render one Flow to a PNG. Returns the output path."""
    _layout(flow)
    nodes_by_id = {n.id: n for n in flow.nodes}

    total_lanes = len(flow.lanes)
    # Width: derive from the rightmost node + padding.
    right_edge = max((n.x + n.w for n in flow.nodes), default=600.0)
    total_w = right_edge + 50
    total_h = total_lanes * LANE_H

    # Figure proportions: scale so each px ≈ 1 fig-unit at the chosen DPI.
    fig_w = max(14.0, total_w / 80.0)
    fig_h = max(5.0, total_h / 80.0 + 1.4)  # extra for title + bottom padding
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(-10, total_w + 10)
    # Reserve 40px above lanes for title; 20 below for bottom labels.
    ax.set_ylim(-50, total_h + 20)
    ax.set_aspect('equal')
    ax.axis('off')
    # BPMN reads top-to-bottom; matplotlib's y grows up. Flip so lane 0 is at top.
    ax.invert_yaxis()

    # Title bar above the lanes
    ax.text(
        total_w / 2, -25, flow.name,
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=TITLE_COLOR,
    )

    for i, lane in enumerate(flow.lanes):
        _draw_lane(ax, lane, i, total_lanes, total_w)

    for n in flow.nodes:
        if n.kind in ('startEvent', 'endEvent'):
            _draw_event(ax, n)
        elif n.kind in ('exclusiveGateway', 'parallelGateway'):
            _draw_gateway(ax, n)
        else:
            _draw_task(ax, n)

    for e in flow.edges:
        _draw_edge(ax, e, nodes_by_id)

    fname = fname or f'{flow.id}.png'
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close(fig)
    print(f'  [OK] {out}')
    return out


def main() -> None:
    print('Rendering business-flow PNGs to', OUT_DIR)
    # Order = same as BPMN viewer dropdown so operator can cross-reference
    for flow_fn in (
        rb.training_flow,           # 1. TrainingFlow
        rb.trading_flow,            # 2. TradingFlow
        rb.risk_gates_flow,         # 3. RiskGates
        rb.models_lifecycle_flow,   # 4. ModelsLifecycle
        rb.registry_claim_flow,     # 5. RegistryClaim
        rb.agent_lifecycle_flow,    # 6. AgentLifecycle
        rb.trainer_dispatch_flow,   # 7. TrainerDispatch
        rb.infra_startup_flow,      # 8. InfraStartup
    ):
        render_flow(flow_fn())
    print('Done.')


if __name__ == '__main__':
    main()
