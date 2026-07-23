"""Render a Value computation graph to a PNG.

If the graphviz ``dot`` binary is installed we use it for a tidy layout;
otherwise we fall back to a pure-matplotlib layered drawing so this course has
no hard dependency on a system package.
"""

from __future__ import annotations

import shutil
from typing import Dict

from .engine import Value, trace


def _has_dot() -> bool:
    return shutil.which("dot") is not None


# --------------------------------------------------------------------- graphviz
def _draw_graphviz(root: Value, path_no_ext: str) -> str:
    from graphviz import Digraph

    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        lbl = n.label or ""
        dot.node(
            uid,
            label=f"{{ {lbl} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            dot.node(uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for a, b in edges:
        dot.edge(str(id(a)), str(id(b)) + b._op)
    out = dot.render(path_no_ext, cleanup=True)
    return out


# ------------------------------------------------------------------- matplotlib
def _longest_depth(root: Value) -> Dict[int, int]:
    """Depth of each node = longest path from any leaf (drives x position)."""
    depth: Dict[int, int] = {}

    def visit(v: Value) -> int:
        if id(v) in depth:
            return depth[id(v)]
        if not v._prev:
            depth[id(v)] = 0
            return 0
        d = 1 + max(visit(c) for c in v._prev)
        depth[id(v)] = d
        return d

    visit(root)
    return depth


def _draw_matplotlib(root: Value, path_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nodes, edges = trace(root)
    depth = _longest_depth(root)

    # group nodes by depth column, assign a row within the column
    cols: Dict[int, list] = {}
    for n in nodes:
        cols.setdefault(depth[id(n)], []).append(n)
    pos: Dict[int, tuple] = {}
    for d, col in cols.items():
        col.sort(key=lambda n: (n.label, n.data))
        for i, n in enumerate(col):
            pos[id(n)] = (d * 3.0, -(i - (len(col) - 1) / 2.0) * 1.6)

    fig_w = max(6, (max(cols) + 1) * 2.6)
    fig_h = max(4, max(len(c) for c in cols.values()) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    box_w, box_h = 2.2, 1.0

    # edges first (so boxes draw on top), with an op tag near the target
    for a, b in edges:
        xa, ya = pos[id(a)]
        xb, yb = pos[id(b)]
        arrow = FancyArrowPatch(
            (xa + box_w / 2, ya),
            (xb - box_w / 2, yb),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#888",
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    for n in nodes:
        x, y = pos[id(n)]
        color = "#f1a7a1" if n._op else "#a8dadc"
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc=color,
            ec="#333",
            lw=1.0,
        )
        ax.add_patch(box)
        head = n.label if n.label else ""
        op = f"  [{n._op}]" if n._op else ""
        ax.text(
            x,
            y + 0.22,
            f"{head}{op}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            x,
            y - 0.05,
            f"data {n.data:.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            x,
            y - 0.28,
            f"grad {n.grad:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#c1121f",
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_title("Computation graph: forward data (blue) and backward grad (red)")
    fig.tight_layout()
    fig.savefig(path_png, dpi=100)
    plt.close(fig)
    return path_png


def draw_graph(root: Value, out_path_no_ext: str) -> str:
    """Render ``root``'s graph to ``<out_path_no_ext>.png``. Returns the file path."""
    if _has_dot():
        return _draw_graphviz(root, out_path_no_ext)
    return _draw_matplotlib(root, out_path_no_ext + ".png")
