"""Exercise (a) — does LoRA rank matter?  r=4 vs r=32.

Fine-tune the SAME model on the SAME data slice twice, changing only the LoRA
rank, then plot the two loss curves together and print the trainable-parameter
counts. Question to answer in your head: does 8x the rank buy you 8x lower loss?
(Spoiler: no — everyday-conversations has low intrinsic rank.)

Runtime on the M5: ~2 short runs, roughly 6-10 min total for 120 steps each.

Fill in the TODOs, then:  uv run python exercises/ex_a_rank.py
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
    # TODO(you): run two fine-tunes that differ ONLY in `rank`.
    # Use run_sft(run_name=..., rank=..., max_steps=STEPS, report_to="none").
    # Keep everything else default. Collect the returned loss history for each.
    hist_low = ...   # TODO(you): rank = 4
    hist_high = ...  # TODO(you): rank = 32

    # TODO(you): plot both loss curves on one axis and save to
    # FIG_DIR / "ex_a_rank_comparison.png". Label each line with its rank.
    ...


if __name__ == "__main__":
    main()
