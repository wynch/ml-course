"""Where this module meets module 04: attention IS a Hopfield retrieval step.

The continuous Hopfield network of Ramsauer et al. (2020) replaces the binary
sign() update with a softmax one:

    ξ_new = Xᵀ · softmax(β · X ξ)

X holds the stored patterns as rows. Line it up against scaled dot-product
attention, softmax(QKᵀ/√d)·V, and the correspondence is exact: the query ξ is
Q, the stored patterns are both K and V, and β plays the part of 1/√d. A
transformer's attention layer is doing associative retrieval from the patterns
in its own context window, once per token, on every forward pass.

Produces figures/attention_hopfield.png:
  top    — a heavily damaged query, retrieved in ONE step at three values of β,
           with the softmax row underneath each: the "attention" over memories;
  bottom — how many patterns each rule can hold. The classical Hebbian net dies
           just above the 0.138·N cliff; the softmax rule keeps going.

Run:  uv run scripts/attention_bridge.py
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
    glyph_patterns,
    hebbian_weights,
    modern_update,
    overlap,
)

CMAP = "cividis"
DAMAGE = 0.35
BETAS_N = [1.0, 4.0, 16.0]   # β·N, so the picture is size-independent


def main() -> None:
    patterns, names = glyph_patterns("ATZLX")
    p, n = patterns.shape
    rng = np.random.default_rng(21)

    target = 2  # the letter Z
    query = corrupt(patterns[target], DAMAGE, rng)
    print(f"query: {names[target]} with {int(DAMAGE * 100)}% of bits flipped, "
          f"overlap {overlap(query, patterns[target]):+.3f}")

    retrieved, weights = [], []
    for bn in BETAS_N:
        r, w = modern_update(patterns, query, beta=bn / n)
        retrieved.append(r)
        weights.append(w)
        print(f"beta*N = {bn:5.1f}  softmax = [" +
              " ".join(f"{name}:{wi:.3f}" for name, wi in zip(names, w)) +
              f"]  overlap(sign(out), {names[target]}) = "
              f"{overlap(np.sign(r), patterns[target]):+.3f}")

    # ---- how far each rule scales ----------------------------------------
    ns = 100
    ps_classic = [5, 10, 15, 20, 30, 50, 80, 130, 200]
    ps_modern = ps_classic + [500, 2000, 5000, 10000, 20000, 40000]
    trials = 25
    classic, modern = [], []
    for pp in ps_modern:
        c_ok, m_ok = 0, 0
        for _ in range(trials):
            pats = rng.choice([-1.0, 1.0], size=(pp, ns))
            mu = int(rng.integers(pp))
            start = corrupt(pats[mu], 0.20, rng)
            if pp in ps_classic:
                W = hebbian_weights(pats)
                fin, _ = async_update(W, start, rng, sweeps=8)
                c_ok += overlap(fin, pats[mu]) >= 0.95
            r, _ = modern_update(pats, start, beta=16.0 / ns)
            m_ok += overlap(np.sign(r), pats[mu]) >= 0.95
        if pp in ps_classic:
            classic.append(c_ok / trials)
        modern.append(m_ok / trials)
        print(f"P = {pp:6d} (alpha = {pp / ns:7.2f})  classical "
              f"{f'{classic[-1]:.2f}' if pp in ps_classic else '   —'}  "
              f"modern {modern[-1]:.2f}")

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(12.4, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.85, 1.5], hspace=0.55, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(as_image(query), cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"query ξ\n{names[target]}, {int(DAMAGE * 100)}% flipped", fontsize=9)

    for i, (bn, r, w) in enumerate(zip(BETAS_N, retrieved, weights)):
        axr = fig.add_subplot(gs[0, i + 1])
        axr.imshow(as_image(r), cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
        axr.set_xticks([]); axr.set_yticks([])
        axr.set_title(rf"$\beta N = {bn:g}$" "\none softmax step", fontsize=9)

        axw = fig.add_subplot(gs[1, i + 1])
        axw.bar(range(p), w, color=["#c1121f" if k == target else "#453781" for k in range(p)])
        axw.set_xticks(range(p))
        axw.set_xticklabels(names, fontsize=8)
        axw.set_ylim(0, 1)
        axw.tick_params(labelsize=7)
        if i == 0:
            axw.set_ylabel("softmax\nweight", fontsize=8)
        axw.set_title(r"$\mathrm{softmax}(\beta X\xi)$", fontsize=8)

    axtxt = fig.add_subplot(gs[1, 0])
    axtxt.axis("off")
    axtxt.text(0.0, 0.95,
               "attention:\n"
               r"  $\mathrm{softmax}(QK^\top\!/\sqrt{d})\,V$" "\n\n"
               "Hopfield:\n"
               r"  $X^\top\mathrm{softmax}(\beta X\xi)$" "\n\n"
               r"$Q=\xi$,  $K=V=X$,  $\beta=1/\sqrt{d}$",
               fontsize=9, va="top", linespacing=1.5)

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(np.array(ps_classic) / ns, classic, "o-", color="#453781", lw=1.8,
             label="classical Hebbian net, asynchronous sweeps")
    ax3.plot(np.array(ps_modern) / ns, modern, "s-", color="#1f918d", lw=1.8,
             label=r"softmax update, one step, $\beta N = 16$")
    ax3.axvline(0.138, color="#c1121f", ls="--", lw=1.3)
    ax3.annotate(r"classical cliff, $\alpha=0.138$", (0.138, 0.90),
                 xytext=(7, 0), textcoords="offset points", color="#c1121f", fontsize=9)
    ax3.set_xscale("log")
    ax3.set_xlabel(r"load  $\alpha = P/N$   (N = 100 neurons, log scale)")
    ax3.set_ylabel("fraction recalled")
    ax3.set_ylim(-0.05, 1.06)
    ax3.set_title("Same memories, two retrieval rules")
    ax3.legend(fontsize=9, loc="lower left")
    ax3.grid(alpha=0.25)

    out = FIGDIR / "attention_hopfield.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
