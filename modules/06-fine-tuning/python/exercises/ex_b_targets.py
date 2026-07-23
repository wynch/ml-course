"""Exercise (b) — WHERE you put the adapter matters as much as how big it is.

Compare two target-module choices at the same rank:
  - attention-only:  ["q_proj", "v_proj"]   (the original LoRA paper's choice)
  - all-linear:      q/k/v/o + gate/up/down  (the modern default)

Train both, overlay the loss curves, and compare trainable-parameter counts.
Which one trains lower? Which one is cheaper? Is the gap worth it here?

Runtime on the M5: ~6-10 min for 120 steps each.

Fill in the TODOs, then:  uv run python exercises/ex_b_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train import run_sft  # noqa: E402

STEPS = 120

ATTN_ONLY = ["q_proj", "v_proj"]
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]


def main():
    # TODO(you): run two fine-tunes differing ONLY in `target_modules`.
    # hist_attn = run_sft(run_name="ex-b-attn", target_modules=ATTN_ONLY, ...)
    # hist_all  = run_sft(run_name="ex-b-all",  target_modules=ALL_LINEAR, ...)
    ...
    # TODO(you): overlay the two loss curves and save to
    # ../figures/ex_b_targets_comparison.png. Note in a comment which config used
    # fewer trainable params (hint: the print_trainable_parameters() lines).
    ...


if __name__ == "__main__":
    main()
