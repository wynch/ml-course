"""Render the module's figures from the JSON traces + eval results.

    uv run python src/figures.py

Produces (into ../figures):
  - handrolled_trace.png : the ReAct timeline of one demo task (blocks:
                           Thought -> Action -> Observation -> ... -> Answer)
  - smolagents_trace.png : the CodeAgent's code-as-action steps
  - eval_success.png     : per-task success rate over N runs (bar chart)
  - eval_steps.png       : distribution of steps-per-run (histogram)

All figures are kept small (<300KB) and use a light, colorblind-friendly palette.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

MODULE_DIR = Path(__file__).resolve().parent.parent.parent
FIG_DIR = MODULE_DIR / "figures"
TRANSCRIPTS = MODULE_DIR / "transcripts"

# kind -> (fill, edge) colors
KIND_COLORS = {
    "thought": ("#e8eef7", "#4577c0"),
    "action": ("#fdeede", "#e08a1e"),
    "observation": ("#e6f4ea", "#2e8b57"),
    "answer": ("#efe6f7", "#7d4fc0"),
    "error": ("#fbe4e4", "#c0392b"),
    "code": ("#fdeede", "#e08a1e"),
}
KIND_LABEL = {
    "thought": "THOUGHT", "action": "ACTION", "observation": "OBSERVATION",
    "answer": "FINAL ANSWER", "error": "PARSE ERROR", "code": "CODE (action)",
}


def _wrap(s: str, width: int, max_lines: int) -> str:
    s = " ".join(s.split()) if "\n" not in s else s
    lines = []
    for para in s.splitlines() or [s]:
        lines.extend(textwrap.wrap(para, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1] + "…"
    return "\n".join(lines)


def _block_timeline(steps, title, out_path, wrap_w=64, keep_newlines=False):
    n = len(steps)
    fig_h = 0.9 + sum(0.55 + 0.20 * min(_est_lines(s, wrap_w, keep_newlines), 6) for s in steps)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)

    y = fig_h - 0.4
    for i, s in enumerate(steps):
        kind = s["kind"]
        fill, edge = KIND_COLORS.get(kind, ("#eee", "#888"))
        body = s["content"]
        text = _wrap(body, wrap_w, 6)
        nlines = text.count("\n") + 1
        h = 0.42 + 0.20 * nlines
        y -= h
        box = FancyBboxPatch((0.4, y), 9.2, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.4, edgecolor=edge, facecolor=fill, mutation_aspect=0.5)
        ax.add_patch(box)
        tag = KIND_LABEL.get(kind, kind.upper())
        if s.get("tool"):
            tag += f"  ·  {s['tool']}"
        ax.text(0.6, y + h - 0.16, tag, fontsize=8.5, fontweight="bold", color=edge, va="top")
        ax.text(0.6, y + h - 0.40, text, fontsize=8.5, family="monospace", color="#222",
                va="top")
        # connector arrow to next
        if i < n - 1:
            ax.annotate("", xy=(5.0, y - 0.02), xytext=(5.0, y + 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.2))
        y -= 0.12

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({out_path.stat().st_size//1024} KB)")


def _est_lines(s, wrap_w, keep_newlines):
    body = s["content"]
    return min(len(_wrap(body, wrap_w, 6).split("\n")), 6)


def handrolled_figure():
    data = json.loads((TRANSCRIPTS / "handrolled_traces.json").read_text())
    # Pick the richest *successful* multi-step trace (most steps, ok=True).
    ok = [t for t in data if t.get("ok")]
    pick = max(ok or data, key=lambda t: len(t["steps"]))
    steps = pick["steps"]
    _block_timeline(steps, "Hand-rolled ReAct loop  ·  " + _wrap(pick["task"], 60, 1),
                    FIG_DIR / "handrolled_trace.png")


def smolagents_figure():
    data = json.loads((TRANSCRIPTS / "smol_traces.json").read_text())
    # Prefer the trace that best shows code-as-action: one snippet calling the
    # MOST distinct tools (e.g. kb_lookup + calculator chained), tie-broken by
    # step count. That is the whole point — a program, not one tool call.
    tool_names = ("kb_lookup", "calculator", "list_module_files")

    def score(t):
        code = " ".join(s["code"] for s in t["steps"])
        distinct = sum(name in code for name in tool_names)
        return (distinct, len(t["steps"]))

    pick = max(data, key=score)
    steps = []
    for s in pick["steps"]:
        if s.get("code"):
            steps.append({"kind": "code", "content": s["code"], "tool": f"step {s['step']}"})
        if s.get("observation"):
            steps.append({"kind": "observation", "content": s["observation"]})
    steps.append({"kind": "answer", "content": pick["answer"]})
    _block_timeline(steps, "smolagents CodeAgent  ·  " + _wrap(pick["task"], 60, 1),
                    FIG_DIR / "smolagents_trace.png", wrap_w=70, keep_newlines=True)


def eval_figures():
    res_path = TRANSCRIPTS / "eval_results.json"
    if not res_path.exists():
        print("no eval_results.json yet — skipping eval figures")
        return
    res = json.loads(res_path.read_text())
    tasks = res["tasks"]
    names = [t["name"] for t in tasks]
    rates = [t["success_rate"] for t in tasks]
    n = res["n_runs"]
    temp = res.get("temperature", 0.5)

    # --- success-rate bar chart ---
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#2e8b57" if r >= 0.8 else "#e08a1e" if r >= 0.4 else "#c0392b" for r in rates]
    bars = ax.bar(range(len(names)), [r * 100 for r in rates], color=colors, edgecolor="#333")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Agent success rate per task  (N={n} runs, temperature {temp:g}, sampling)",
                 fontweight="bold")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r * 100 + 2, f"{r*100:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(100, ls=":", color="#999", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_success.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'eval_success.png'}")

    # --- steps-per-run histogram ---
    all_steps = [s for t in tasks for s in t["steps_per_run"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo, hi = min(all_steps), max(all_steps)
    bins = range(lo, hi + 2)
    ax.hist(all_steps, bins=bins, color="#4577c0", edgecolor="#222", align="left", rwidth=0.85)
    ax.set_xlabel("steps taken to finish (or hit max_steps)")
    ax.set_ylabel("count of runs")
    ax.set_xticks(list(range(lo, hi + 1)))
    ax.set_title(f"How many steps did the agent take?  (all {len(all_steps)} runs)",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_steps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'eval_steps.png'}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    handrolled_figure()
    if (TRANSCRIPTS / "smol_traces.json").exists():
        smolagents_figure()
    eval_figures()


if __name__ == "__main__":
    main()
