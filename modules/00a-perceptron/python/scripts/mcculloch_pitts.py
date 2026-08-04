"""Logic with a threshold — the 1943 neuron, no learning involved.

Prints truth tables for hand-wired gates and draws where each gate's threshold
line falls in the unit square.

Produces:
  figures/mcculloch_pitts.png

Run:  uv run scripts/mcculloch_pitts.py
"""

import _bootstrap  # noqa: F401  (sets sys.path + FIGDIR)
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perceptron.mp_neuron import GATES, INPUTS2, truth_table, xor_from_gates


def main() -> None:
    for name in ("AND", "OR", "NAND", "NOR"):
        w, theta = GATES[name]
        rows = truth_table(name)
        table = "  ".join(f"{a}{b}→{o}" for a, b, o in rows)
        print(f"{name:5s} w={w} θ={theta:+.1f}   {table}")

    xor_rows = [(a, b, xor_from_gates((a, b))) for a, b in INPUTS2]
    print("XOR   AND(OR, NAND) — two layers   "
          + "  ".join(f"{a}{b}→{o}" for a, b, o in xor_rows))
    expected = [0, 1, 1, 0]
    assert [r[2] for r in xor_rows] == expected, xor_rows
    print("XOR   matches the truth table exactly (needs a hidden layer).")

    # ---- figure: each gate's threshold line across the unit square ---------
    names = ["AND", "OR", "NAND", "NOR"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5), sharex=True, sharey=True)
    xs = np.linspace(-0.45, 1.45, 50)
    for ax, name in zip(axes, names):
        (w1, w2), theta = GATES[name]
        rows = truth_table(name)
        on = np.array([[a, b] for a, b, o in rows if o == 1], dtype=float)
        off = np.array([[a, b] for a, b, o in rows if o == 0], dtype=float)
        # shade the firing half-plane
        XX, YY = np.meshgrid(np.linspace(-0.45, 1.45, 220),
                             np.linspace(-0.45, 1.45, 220))
        fires = (w1 * XX + w2 * YY >= theta).astype(float)
        ax.contourf(XX, YY, fires, levels=[-0.5, 0.5, 1.5],
                    colors=["#f2f4f3", "#cfe6e5"])
        ax.plot(xs, (theta - w1 * xs) / w2, color="#1f918d", lw=2)
        if len(on):
            ax.scatter(on[:, 0], on[:, 1], s=95, c="#1f918d",
                       edgecolors="white", zorder=3, label="fires (1)")
        if len(off):
            ax.scatter(off[:, 0], off[:, 1], s=95, c="#453781",
                       edgecolors="white", zorder=3, label="silent (0)")
        ax.set_title(f"{name}   w={list(GATES[name][0])}, θ={theta:+.0f}", fontsize=9)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_aspect("equal")
    axes[0].set_ylabel("x₂")
    for ax in axes:
        ax.set_xlabel("x₁")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("McCulloch–Pitts gates: one line, chosen by hand", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mcculloch_pitts.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'mcculloch_pitts.png'}")


if __name__ == "__main__":
    main()
