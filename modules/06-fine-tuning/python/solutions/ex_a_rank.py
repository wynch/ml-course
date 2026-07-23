"""Solution (a) — LoRA rank r=4 vs r=32.  Runs ~6-10 min on the M5.

Trains two adapters differing only in rank, overlays their loss curves, and
writes ../figures/ex_a_rank_comparison.png. The takeaway: r=4 already captures
almost all the gain on this small, low-diversity dataset — rank buys capacity
you don't need here, at 8x the trainable parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
STEPS = 120


def main():
    hist_low, _ = run_sft(
        run_name="ex-a-r4", rank=4, max_steps=STEPS, report_to="none"
    )
    hist_high, _ = run_sft(
        run_name="ex-a-r32", rank=32, max_steps=STEPS, report_to="none"
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for hist, label, color in [
        (hist_low, "r=4  (1.1M trainable)", "#2563eb"),
        (hist_high, "r=32 (17.4M trainable)", "#e11d48"),
    ]:
        steps = [h["step"] for h in hist]
        loss = [h["loss"] for h in hist]
        ax.plot(steps, loss, lw=2, marker="o", ms=3, label=label, color=color)

    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title("LoRA rank: r=4 vs r=32 — 8x the params, marginal loss gain")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "ex_a_rank_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

    if hist_low and hist_high:
        print(f"r=4  final loss: {hist_low[-1]['loss']:.4f}")
        print(f"r=32 final loss: {hist_high[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
