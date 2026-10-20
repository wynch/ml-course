"""Solution (a) — LoRA rank r=4 vs r=32.  Runs ~6-10 min on the M5.

Trains two adapters differing only in rank, overlays their loss curves, and
writes ../figures/ex_a_rank_comparison.png. The result: the same final training
loss (0.6549 vs 0.6553) for 8x the trainable parameters, and the same held-out
loss (eval_heldout.py). Two caveats before reading it as "rank does not
matter": both runs keep lora_alpha=32, so the update scale alpha/r is 8 at r=4
and 1 at r=32, which makes r=32 take much smaller steps; and training loss on
data seen about twice says little about the task. A fairer test sets
lora_alpha=2*r for both, or use_rslora=True (scale alpha/sqrt(r)).

  uv run python solutions/ex_a_rank.py --plot-only   # redraw from results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"
STEPS = 120


def main():
    if "--plot-only" in sys.argv:
        hist_low = json.loads((RESULTS / "ex-a-r4_loss_history.json").read_text())
        hist_high = json.loads((RESULTS / "ex-a-r32_loss_history.json").read_text())
    else:
        hist_low, _ = run_sft(
            run_name="ex-a-r4", rank=4, max_steps=STEPS, report_to="none"
        )
        hist_high, _ = run_sft(
            run_name="ex-a-r32", rank=32, max_steps=STEPS, report_to="none"
        )

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for hist, label, color in [
        (hist_low, "r=4  (2.2M trainable, α/r = 8)", "#2563eb"),
        (hist_high, "r=32 (17.4M trainable, α/r = 1)", "#e11d48"),
    ]:
        steps = [h["step"] for h in hist]
        loss = [h["loss"] for h in hist]
        ax.plot(steps, loss, lw=2, marker="o", ms=3, label=label, color=color)

    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title("LoRA rank: r=4 vs r=32 — 8x the params, same training loss")
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
