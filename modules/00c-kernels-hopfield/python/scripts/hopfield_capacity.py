"""How many memories fit? Measure the 0.138·N capacity instead of quoting it.

For N ∈ {100, 200, 400, 800, 1600} and loads α = P/N from 0.04 to 0.28, store P
random ±1 patterns, hand the network one of them with 5% of its bits flipped,
run it to a fixed point, and record whether it came back (overlap ≥ 0.95).

The statistical-mechanics answer (Amit, Gutfreund & Sompolinsky, 1985) is that
retrieval states stop existing above α_c ≈ 0.138 in the N → ∞ limit. At the
finite sizes you can simulate on a laptop the collapse sits higher and drifts
downwards as N grows, which is what the second panel plots.

Produces figures/hopfield_capacity.png. Takes ~45 s.

Run:  uv run scripts/hopfield_capacity.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernelmem.hopfield import capacity_curve, critical_alpha

SIZES = [100, 200, 400, 800, 1600]
ALPHAS = np.round(np.arange(0.04, 0.2801, 0.005), 4)
TRIALS = 40
THEORY = 0.138


def main() -> None:
    results = {}
    for n in SIZES:
        alphas, mean_m, frac = capacity_curve(n=n, alphas=ALPHAS, trials=TRIALS, seed=11)
        ac = critical_alpha(alphas, frac)
        results[n] = (alphas, mean_m, frac, ac)
        print(f"N = {n:5d}  P at alpha_c = {ac * n:6.1f}  alpha_c(50% recall) = {ac:.4f}"
              f"   mean overlap at alpha = 0.138: {np.interp(THEORY, alphas, mean_m):.3f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.55, 1.0]})
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(SIZES)))

    for (n, (alphas, _m, frac, ac)), colour in zip(results.items(), colours):
        ax.plot(alphas, frac, color=colour, lw=1.8, label=f"N = {n}")
        ax.plot([ac], [0.5], marker="o", ms=5, color=colour)
    ax.axvline(THEORY, color="#c1121f", ls="--", lw=1.4)
    ax.annotate(r"theory: $\alpha_c \approx 0.138$", (THEORY, 0.06),
                xytext=(6, 0), textcoords="offset points", color="#c1121f", fontsize=9)
    ax.set_xlabel(r"load  $\alpha = P/N$")
    ax.set_ylabel("fraction recalled (overlap ≥ 0.95)")
    ax.set_title("The memory cliff: recall collapses at a critical load")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    ns = np.array(SIZES, dtype=float)
    acs = np.array([results[n][3] for n in SIZES])
    ax2.plot(1.0 / ns, acs, "o-", color="#1f918d", lw=1.8)
    for n, ac in zip(SIZES, acs):
        ax2.annotate(f"N={n}", (1.0 / n, ac), textcoords="offset points",
                     xytext=(6, 5), fontsize=8, color="#5c6b67")
    ax2.axhline(THEORY, color="#c1121f", ls="--", lw=1.4)
    ax2.annotate("0.138", (0.0, THEORY), xytext=(4, -13), textcoords="offset points",
                 color="#c1121f", fontsize=9)
    ax2.set_xlabel("1 / N")
    ax2.set_ylabel(r"measured $\alpha_c$")
    ax2.set_title("Finite-size drift towards the theoretical limit")
    ax2.set_xlim(-0.0006, 1.0 / min(SIZES) * 1.08)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = FIGDIR / "hopfield_capacity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # a headline number for the README: how many letters fit in N = 100
    print(f"at N = 100, 0.138·N = {0.138 * 100:.1f} random patterns — "
          f"measured collapse at P = {results[100][3] * 100:.1f}")


if __name__ == "__main__":
    main()
