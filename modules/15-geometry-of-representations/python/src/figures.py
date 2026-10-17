"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
(manifold.png reads FashionMNIST to draw its images; the Hub cache serves it.)
"""

from __future__ import annotations

from common import FIGURES, read_results

import cka
import intrinsic_dim
import manifold
import regions


def main() -> None:
    regions.fig_regions(read_results("regions"), FIGURES / "regions.png")
    intrinsic_dim.fig_id(read_results("intrinsic_dim"), FIGURES / "intrinsic_dim.png")
    manifold.fig_manifold(read_results("manifold"), FIGURES / "manifold.png")
    cka.fig_cka(read_results("cka"), FIGURES / "cka.png")


if __name__ == "__main__":
    main()
