"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

from common import FIGURES, read_results

import gpt2_sae
import sae
import steer
import toy


def main() -> None:
    toy.fig_toy(read_results("toy"), FIGURES / "toy_superposition.png")
    s = read_results("sae")
    sae.fig_sae_metrics(s, FIGURES / "sae_metrics.png")
    sae.fig_sae_features(s, FIGURES / "sae_features.png")
    gpt2_sae.fig_gpt2(read_results("gpt2_sae"), FIGURES / "gpt2_sae.png")
    steer.fig_steer(read_results("steer"), FIGURES / "steer.png")


if __name__ == "__main__":
    main()
