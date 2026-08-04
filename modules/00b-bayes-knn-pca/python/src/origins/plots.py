"""Shared matplotlib setup for module 00b's figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Track 0 amber, used for the "our implementation" series everywhere.
T0 = "#a86b0e"
ACCENT = "#1f918d"
WARN = "#c0504d"
INK = "#1c2422"
MUTED = "#5c6b67"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#8d9a96",
        "axes.grid": True,
        "grid.color": "#d7dedb",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


def finish(fig, path, title=None):
    """Optionally title, save and close a figure; print where it went."""
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")
