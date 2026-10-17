"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

from common import FIGURES, read_results

import bt_synthetic
import dpo_train
import length_bias
import reward_head


def main() -> None:
    bt_synthetic.fig_bt(read_results("bt_synthetic"), FIGURES / "bt_synthetic.png")
    reward_head.fig_reward(read_results("reward_head"), FIGURES / "reward_head.png")
    for cfg in ("preview", "full"):
        dpo_train.fig_dpo(read_results(f"dpo_{cfg}"), FIGURES / f"dpo_{cfg}.png")
    length_bias.fig_length(read_results("length_bias"), FIGURES / "length_bias.png")


if __name__ == "__main__":
    main()
