"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

from common import FIGURES, read_results

import compare
import decode
import gru
import hf_compare
import ssm
import window
import window_cost


def main() -> None:
    gru.fig_gru(read_results("gru"), FIGURES / "gru.png")
    ssm.fig_ssm(read_results("ssm"), FIGURES / "ssm.png")
    compare.fig_compare(read_results("compare"), FIGURES / "compare.png")
    decode.fig_decode(read_results("decode"), FIGURES / "decode.png")
    hf_compare.fig_hf(read_results("hf_compare"), FIGURES / "hf_compare.png")
    window.fig_window(read_results("window"), FIGURES / "window.png")
    window_cost.fig_cost(read_results("window_cost"), FIGURES / "window_cost.png")


if __name__ == "__main__":
    main()
