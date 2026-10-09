"""Store five 10×10 letters in a Hopfield net and watch it repair damage.

Produces:
  figures/hopfield_retrieval.png — a row per letter: the stored pattern, the
      corrupted input, three snapshots of the asynchronous sweep, the result.
  figures/hopfield_energy.png    — the energy trace of those runs, one line per
      letter, next to the energy of every stored pattern.

Run:  uv run scripts/hopfield_demo.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernelmem.hopfield import (
    as_image,
    async_update,
    corrupt,
    energy,
    glyph_patterns,
    hebbian_weights,
    overlap,
)

CORRUPT = 0.30
SEED = 4
CMAP = "cividis"


def main() -> None:
    patterns, names = glyph_patterns("ATZLX")
    p, n = patterns.shape
    W = hebbian_weights(patterns)
    print(f"stored {p} patterns of N = {n} neurons  → load alpha = {p / n:.3f}")

    ov = patterns @ patterns.T / n
    off = ov[~np.eye(p, dtype=bool)]
    print(f"pairwise overlap between stored letters: min {off.min():+.2f} "
          f"max {off.max():+.2f} (0 would be ideal; letters are correlated)")
    for name, xi in zip(names, patterns):
        print(f"  E({name}) = {energy(W, xi):+.3f}")

    rng = np.random.default_rng(SEED)
    runs = []
    for i, name in enumerate(names):
        start = corrupt(patterns[i], CORRUPT, rng)
        final, trace = async_update(W, start, rng, sweeps=6, record=True)
        runs.append((name, patterns[i], start, final, trace))
        print(f"{name}: overlap start {overlap(start, patterns[i]):+.3f} "
              f"→ final {overlap(final, patterns[i]):+.3f}   "
              f"E {trace[0][1]:+.3f} → {trace[-1][1]:+.3f}   "
              f"({len(trace) - 1} neuron updates)")

    # ---- figure 1: the retrieval strip ------------------------------------
    snap_at = [10, 25, 50, 80]
    cols = 2 + len(snap_at) + 1
    fig, axes = plt.subplots(len(runs), cols, figsize=(1.35 * cols, 1.42 * len(runs)))
    for r, (name, target, start, final, trace) in enumerate(runs):
        panels = [("stored", target), (f"−{int(CORRUPT * 100)}% bits", start)]
        for k in snap_at:
            k = min(k, len(trace) - 1)
            panels.append((f"t={trace[k][0]}", trace[k][2]))
        panels.append(("settled", final))
        for c, (title, vec) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(as_image(vec), cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=12, rotation=0, labelpad=10, va="center")
            if c == cols - 1:
                m = overlap(vec, target)
                ax.set_xlabel(f"m={m:+.2f}", fontsize=8,
                              color="#1f918d" if m > 0.95 else "#c1121f")
    fig.suptitle("Content-addressable memory: damage in, stored pattern out", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "hopfield_retrieval.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- figure 2: energy descent -----------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                  gridspec_kw={"width_ratios": [1.6, 1.0]})
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.86, len(runs)))
    for (name, target, _s, _f, trace), colour in zip(runs, colours):
        ks = [t[0] for t in trace]
        es = [t[1] for t in trace]
        ax.step(ks, es, where="post", color=colour, lw=1.6, label=f"recall of {name}")
        ax.axhline(energy(W, target), color=colour, ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel("neuron updates (asynchronous, random order)")
    ax.set_ylabel(r"energy  $E(s) = -\frac{1}{2}\, s^\top W s$")
    ax.set_title("Every accepted flip lowers E — the state slides into a valley")
    ax.set_xlim(0, 120)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # a monotonicity receipt
    worst = max(max(np.diff([t[1] for t in trace])) for *_x, trace in runs)
    print(f"largest single-update energy INCREASE across all runs: {worst:+.3e} "
          "(should be ≤ 0 up to float noise)")

    starts = [t[0][1] for *_x, t in runs]
    ends = [t[-1][1] for *_x, t in runs]
    idx = np.arange(len(runs))
    ax2.bar(idx - 0.2, starts, width=0.4, color="#c1121f", label="corrupted input")
    ax2.bar(idx + 0.2, ends, width=0.4, color="#1f918d", label="settled state")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("energy")
    ax2.set_title("Where each run started and stopped")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = FIGDIR / "hopfield_energy.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
