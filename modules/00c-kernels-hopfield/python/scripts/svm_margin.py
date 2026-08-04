"""Fit the hard-margin SVM in the dual and draw the margin it found.

Produces figures/svm_margin.png:
  left  — the data, the separating line, the two margin lines, the support
          vectors ringed, and every αᵢ printed next to its point;
  right — the dual objective W(α) climbing under projected gradient ascent,
          and the α spectrum showing how few entries are non-zero.

Run:  uv run scripts/svm_margin.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernelmem.svm import DualSVM, separable_blobs

NEG = "#453781"   # viridis-dark  → class −1
POS = "#5ec962"   # viridis-light → class +1
LINE = "#1c2422"


def main() -> None:
    X, y = separable_blobs()
    model = DualSVM(kernel="linear", iters=4000).fit(X, y)

    w = model.weights()
    sv = model.support_mask()
    fmargin = y * model.decision_function(X)

    print(f"n = {len(y)} points, {int(sv.sum())} support vectors")
    print(f"w = [{w[0]:.6f}, {w[1]:.6f}]   b = {model.b:.6f}")
    print(f"margin 2/||w||      = {2.0 / np.linalg.norm(w):.6f}")
    print(f"margin 2/sqrt(sum a)= {model.margin():.6f}   (equal at the optimum)")
    print(f"sum(alpha) = {model.alpha.sum():.6f}   sum(alpha*y) = {model.alpha @ y:.2e}")
    print(f"smallest functional margin y*f(x) = {fmargin.min():.6f}  (should be 1)")
    print("support vectors:")
    for i in np.where(sv)[0]:
        print(f"  point {i:2d}  y = {y[i]:+.0f}  alpha = {model.alpha[i]:.6f}  "
              f"x = ({X[i, 0]:+.3f}, {X[i, 1]:+.3f})")

    # ---- figure -----------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.0),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})

    pad = 0.9
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200)

    def line(offset):
        return -(w[0] * xs + model.b - offset) / w[1]

    ax.fill_between(xs, line(-1), line(1), color="#1f918d", alpha=0.10, lw=0,
                    label="the margin band")
    ax.plot(xs, line(0), color=LINE, lw=2.0, label=r"$w^\top x + b = 0$")
    ax.plot(xs, line(1), color=LINE, lw=1.0, ls="--")
    ax.plot(xs, line(-1), color=LINE, lw=1.0, ls="--")

    for cls, colour, name in ((-1, NEG, "class −1"), (1, POS, "class +1")):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=58, c=colour, edgecolors="white",
                   linewidths=1.0, zorder=3, label=name)
    ax.scatter(X[sv, 0], X[sv, 1], s=230, facecolors="none", edgecolors="#c1121f",
               linewidths=2.0, zorder=4, label="support vectors")
    for i in np.where(sv)[0]:
        dx = 14 if y[i] > 0 else -14
        ax.annotate(rf"$\alpha={model.alpha[i]:.3f}$", (X[i, 0], X[i, 1]),
                    textcoords="offset points", xytext=(dx, -22), fontsize=9,
                    color="#c1121f", ha="left" if y[i] > 0 else "right")

    # the margin as a measured arrow, perpendicular to the boundary, drawn up
    # in the empty part of the plot where it does not sit on top of any point
    unit = w / np.linalg.norm(w)
    anchor_y = X[:, 1].min() + 0.30
    mid = np.array([-(w[1] * anchor_y + model.b) / w[0], anchor_y])
    ax.annotate("", xy=mid + unit * model.margin() / 2, xytext=mid - unit * model.margin() / 2,
                arrowprops=dict(arrowstyle="<->", color="#1f918d", lw=1.8))
    ax.annotate(rf"$\frac{{2}}{{\|w\|}} = {model.margin():.3f}$",
                mid + unit * model.margin() / 2, textcoords="offset points",
                xytext=(2, -26), color="#1f918d", fontsize=11, ha="center")

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Hard-margin SVM · {int(sv.sum())} of {len(y)} points decide the boundary")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    # ---- right: the ascent and the sparsity -------------------------------
    steps = [h[0] for h in model.history]
    obj = [h[1] for h in model.history]
    ax2.plot(steps, obj, color="#1f918d", lw=2)
    ax2.set_xscale("symlog", linthresh=10)
    ax2.set_xlabel("projected-gradient-ascent step")
    ax2.set_ylabel(r"dual objective  $W(\alpha)$")
    ax2.set_title(rf"$W(\alpha)\to{obj[-1]:.4f}$, and $\|w\|^2=\sum_i\alpha_i$")
    ax2.grid(alpha=0.25)

    inset = ax2.inset_axes([0.42, 0.16, 0.55, 0.42])
    order = np.argsort(-model.alpha)
    inset.bar(range(len(y)), model.alpha[order],
              color=["#c1121f" if sv[i] else "#9daba7" for i in order])
    inset.set_title(r"every $\alpha_i$, sorted", fontsize=8)
    inset.set_xlabel("point", fontsize=7)
    inset.tick_params(labelsize=7)

    fig.tight_layout()
    out = FIGDIR / "svm_margin.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
