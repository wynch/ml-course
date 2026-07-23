"""Concept figures for module 06 — no training, just intuition.

Produces two PNGs in ../figures/:
  1. param_count_bars.png  — full fine-tune vs LoRA trainable-parameter counts
     at r = 8 / 16 / 32, with the "you train 0.x%" punchline. The LoRA numbers
     are REAL: we actually wrap SmolLM2-360M with peft and count.
  2. lora_in_attention.png — a schematic of where the LoRA A·B adapter sits
     next to a frozen projection weight inside the attention block.

Run:  uv run python figures_concept.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from src.common import LORA_ALPHA, LORA_TARGETS, MODEL_ID, count_params

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_FROZEN = "#94a3b8"   # slate — frozen weights
C_TRAIN = "#e11d48"    # rose  — the tiny trainable slice
C_A = "#2563eb"        # blue  — LoRA A
C_B = "#16a34a"        # green — LoRA B


def measure_lora_counts(ranks=(8, 16, 32)):
    """Load the base model once, wrap at each rank, count trainable params."""
    rows = []
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    total_base, _ = count_params(base)
    for r in ranks:
        cfg = LoraConfig(
            r=r,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        # get_peft_model wraps a fresh copy each call via the same base weights;
        # we re-wrap the ORIGINAL base each time by reloading the adapter layer.
        peft_model = get_peft_model(
            AutoModelForCausalLM.from_pretrained(MODEL_ID), cfg
        )
        total, trainable = count_params(peft_model)
        rows.append((r, total, trainable))
        del peft_model
    return total_base, rows


def fig_param_bars():
    total_base, rows = measure_lora_counts()

    labels = ["Full\nfine-tune"] + [f"LoRA\nr={r}" for r, _, _ in rows]
    # Full fine-tune trains every parameter; LoRA trains only the adapters.
    trainables = [total_base] + [tr for _, _, tr in rows]
    totals = [total_base] + [tot for _, tot, _ in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(labels))

    # frozen (grey) stacked under trainable (rose) — full bar height = total params
    frozen = [tot - tr for tot, tr in zip(totals, trainables)]
    ax.bar(x, frozen, color=C_FROZEN, label="frozen params")
    ax.bar(x, trainables, bottom=frozen, color=C_TRAIN, label="trainable params")

    for xi, (tr, tot) in enumerate(zip(trainables, totals)):
        pct = 100 * tr / tot
        ax.text(
            xi,
            tot + total_base * 0.015,
            f"{tr/1e6:.2f}M\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=C_TRAIN if pct < 50 else "#334155",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("parameters")
    ax.set_ylim(0, total_base * 1.30)
    ax.set_title(
        f"{MODEL_ID.split('/')[-1]} — you train a sliver of the model\n"
        f"total = {total_base/1e6:.0f}M params",
        fontsize=11,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.0f}M")

    fig.tight_layout()
    out = FIG_DIR / "param_count_bars.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    # echo the punchline numbers for the README / report
    for r, tot, tr in rows:
        print(f"  LoRA r={r:>2}: {tr/1e6:.3f}M trainable of {tot/1e6:.1f}M  = {100*tr/tot:.3f}%")


def fig_lora_schematic():
    """Where the adapter sits: h = W0 x + (alpha/r) * B(A x), with W0 frozen."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, color, text, tc="white", fs=10, alpha=1.0):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=color, edgecolor="none", alpha=alpha,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold")

    def arrow(x0, y0, x1, y1, color="#334155"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    # input
    box(0.3, 2.6, 1.1, 0.9, "#334155", "x\n(hidden)", fs=10)

    # frozen projection W0 (top path)
    box(3.0, 4.3, 2.4, 1.0, C_FROZEN, "W₀  (frozen)\nq/k/v/o proj", tc="#1e293b")
    # LoRA path (bottom)
    box(2.7, 0.9, 1.3, 0.9, C_A, "A\n(d×r)")
    box(4.6, 0.9, 1.3, 0.9, C_B, "B\n(r×d)")
    ax.text(3.35, 0.35, "r ≪ d", ha="center", fontsize=9, color=C_A, style="italic")

    # sum node
    ax.add_patch(mpatches.Circle((7.2, 2.9), 0.35, facecolor="#0f172a"))
    ax.text(7.2, 2.9, "+", ha="center", va="center", color="white",
            fontsize=16, fontweight="bold")

    # output
    box(8.4, 2.45, 1.3, 0.9, "#334155", "h", fs=12)

    # wires
    arrow(1.4, 3.05, 3.0, 4.8)          # x -> W0
    arrow(1.4, 3.05, 2.7, 1.35)         # x -> A
    arrow(4.0, 1.35, 4.6, 1.35)         # A -> B
    arrow(5.4, 4.8, 7.0, 3.15)          # W0 -> +
    arrow(5.9, 1.35, 7.0, 2.65)         # B -> +
    arrow(7.55, 2.9, 8.4, 2.9)          # + -> h

    ax.text(6.15, 1.55, "× (α/r)", fontsize=9, color=C_B, style="italic")
    ax.text(5.0, 5.55,
            "h = W₀·x  +  (α/r)·B(A·x)",
            ha="center", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(5.0, 0.05,
            "Only A and B are trained. W₀ never moves. A starts random, B starts at 0,\n"
            "so the adapter is a no-op at step 0 and the model begins exactly as the base.",
            ha="center", fontsize=8.5, color="#475569")

    fig.tight_layout()
    out = FIG_DIR / "lora_in_attention.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_param_bars()
    fig_lora_schematic()
