"""The headline run: train the perceptron and compare mistakes to the bound.

Prints the final weights, the mistake count, R, γ and Novikoff's (R/γ)²; writes
a JSON blob of the run so the Zig lane, the tests, and the explorable can all
check themselves against the same numbers.

Produces:
  figures/perceptron_evolution.png   the line snapping to place, 6 panels
  figures/mistakes_vs_bound.png      cumulative mistakes vs the guarantee
  run_perceptron.json                the run's numbers, for cross-checking

Run:  uv run scripts/train_perceptron.py
"""

import json
import pathlib

import _bootstrap  # noqa: F401  (sets sys.path + FIGDIR)
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perceptron import rosenblatt
from perceptron.data import SEED, augment, separable_2d

N = 200
MARGIN = 0.35
POS = "#1f918d"
NEG = "#453781"
OUT = pathlib.Path(__file__).resolve().parents[1] / "run_perceptron.json"


def scatter(ax, X, y):
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=13, c=POS, label="+1", zorder=2)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=13, c=NEG, label="−1", zorder=2)


def main() -> None:
    X, y = separable_2d(n=N, margin=MARGIN, seed=SEED)
    Xa = augment(X)
    run = rosenblatt.train(Xa, y, max_epochs=100)
    info = rosenblatt.novikoff_bound(Xa, y)

    acc = run.accuracy(Xa, y)
    print(f"data      n={N}  margin≥{MARGIN}  seed={SEED}")
    print(f"final w   [{run.w[0]:.6f}, {run.w[1]:.6f}, {run.w[2]:.6f}]")
    print(f"mistakes  {run.mistakes}   epochs {run.epochs}   "
          f"converged {run.converged}   accuracy {acc*100:.1f}%")
    print(f"per-epoch mistakes {run.per_epoch}")
    print(f"R         {info['R']:.6f}   (augmented radius, max‖[x,1]‖)")
    print(f"γ         {info['gamma']:.6f}   (margin achieved by the Frank–Wolfe separator)")
    print(f"          hull distance {info['gamma_hull']:.6f} — γ* is bracketed to "
          f"{info['gap']:.2e}")
    print(f"bound     (R/γ)² = {info['bound']:.2f}")
    print(f"ratio     empirical / bound = {run.mistakes / info['bound']:.4f}")
    assert run.converged and acc == 1.0
    assert run.mistakes <= info["bound"], "Novikoff violated — check the maths!"

    # ---- figure 1: the line snapping into place ---------------------------
    picks = sorted({0, 1, 2, round(run.mistakes / 3), round(2 * run.mistakes / 3),
                    run.mistakes})
    for k in range(run.mistakes + 1):          # top up if duplicates collapsed it
        if len(picks) >= 6:
            break
        if k not in picks:
            picks = sorted(picks + [k])
    picks = picks[:6]
    xs = np.linspace(-3.2, 3.2, 40)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), picks):
        w = run.snapshots[k]
        scatter(ax, X, y)
        pred = np.where(Xa @ w > 0, 1, -1) if np.any(w) else np.zeros(len(y))
        wrong = pred != y
        ax.scatter(X[wrong, 0], X[wrong, 1], s=52, facecolors="none",
                   edgecolors="#c1121f", lw=0.9, zorder=3)
        ys = rosenblatt.line_from_weights(w, xs)
        if not np.all(np.isnan(ys)):
            ax.plot(xs, ys, color="#c1121f", lw=2, zorder=4)
        elif abs(w[0]) > 1e-12:
            ax.axvline(-w[2] / w[0], color="#c1121f", lw=2, zorder=4)
        ax.set_title(f"after {k} mistake{'' if k == 1 else 's'} · "
                     f"{int(wrong.sum())} misclassified", fontsize=9)
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-3.3, 3.3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Rosenblatt's rule: w ← w + y·x, one update per mistake "
        f"({run.mistakes} in total, {N} points, margin ≥ {MARGIN})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "perceptron_evolution.png", dpi=110)
    plt.close(fig)

    # ---- figure 2: mistakes vs the bound ----------------------------------
    cum = np.asarray(run.trace)
    steps = np.arange(len(cum))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.step(steps, cum, where="post", color="#1f918d", lw=2,
            label=f"empirical mistakes (final {run.mistakes})")
    ax.axhline(info["bound"], color="#c1121f", lw=2, ls="--",
               label=f"Novikoff bound (R/γ)² = {info['bound']:.1f}")
    for ep in range(1, run.epochs):
        ax.axvline(ep * N, color="#5c6b67", lw=0.7, ls=":", alpha=0.7)
    ax.text(N * 0.02, info["bound"] * 0.94,
            f"R = {info['R']:.3f}   γ = {info['gamma']:.4f}",
            fontsize=9, color="#5c6b67", va="top")
    ax.set_xlabel("examples processed (dotted lines = epoch boundaries)")
    ax.set_ylabel("cumulative mistakes")
    ax.set_title("The guarantee is real but loose: "
                 f"{run.mistakes} mistakes against a bound of {info['bound']:.0f}")
    ax.set_ylim(0, info["bound"] * 1.12)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mistakes_vs_bound.png", dpi=110)
    plt.close(fig)

    # ---- the run, as JSON, for the other lanes ----------------------------
    blob = {
        "seed": SEED, "n": N, "margin": MARGIN,
        "w": [float(v) for v in run.w],
        "mistakes": run.mistakes, "epochs": run.epochs,
        "per_epoch": run.per_epoch, "accuracy": acc,
        "R": info["R"], "gamma": info["gamma"], "gamma_hull": info["gamma_hull"],
        "bound": info["bound"],
        "u": [float(v) for v in info["u"]],
        "first_updates": [
            {"mistake": m, "index": i, "w": [float(v) for v in run.snapshots[m]]}
            for m, i in run.updates[:12]
        ],
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"wrote {FIGDIR / 'perceptron_evolution.png'}")
    print(f"wrote {FIGDIR / 'mistakes_vs_bound.png'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
