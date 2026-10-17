"""Redraw every README figure from the committed results/*.json, no model needed.

    uv run python src/figures.py

To recompute the numbers too, run each script with --figures instead.
"""

from __future__ import annotations

from common import FIGURES, read_results

import hooks
import patching
import poles
import probe


def main() -> None:
    hooks.fig_hook(read_results("hooks"), FIGURES / "hook_neuron.png")
    probe.fig_probe(read_results("probe"), FIGURES / "probe_by_layer.png")
    patching.fig_patching(read_results("patching"), FIGURES / "patching_heatmap.png")
    poles.fig_poles(read_results("poles"), FIGURES / "neuron_poles.png")


if __name__ == "__main__":
    main()
