"""Solution (b) — attention-only q/v vs all-linear targets. Runs ~6-10 min.

Trains two adapters at the same rank differing only in target_modules, overlays
the loss curves, and writes ../figures/ex_b_targets_comparison.png. Takeaway:
all-linear reaches a slightly lower loss (the MLP projections carry a lot of the
model's behaviour) but costs more trainable params; q/v-only is a leaner steer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
STEPS = 120

ATTN_ONLY = ["q_proj", "v_proj"]
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]


def main():
    hist_attn, _ = run_sft(
        run_name="ex-b-attn", target_modules=ATTN_ONLY,
        max_steps=STEPS, report_to="none",
    )
    hist_all, _ = run_sft(
        run_name="ex-b-all", target_modules=ALL_LINEAR,
        max_steps=STEPS, report_to="none",
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for hist, label, color in [
        (hist_attn, "q_proj + v_proj only", "#2563eb"),
        (hist_all, "all linear (q/k/v/o + MLP)", "#e11d48"),
    ]:
        steps = [h["step"] for h in hist]
        loss = [h["loss"] for h in hist]
        ax.plot(steps, loss, lw=2, marker="o", ms=3, label=label, color=color)

    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title("Where the adapter lives: attention-only vs all-linear")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "ex_b_targets_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    if hist_attn and hist_all:
        print(f"q/v-only  final loss: {hist_attn[-1]['loss']:.4f}")
        print(f"all-linear final loss: {hist_all[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
