"""Least squares solved twice — by equation and by gradient descent.

Fits the same noisy line both ways, checks that they agree, and draws the two
pictures that explain what the fit *is*: a right angle in n-dimensional space,
and a walk downhill on a bowl.

Produces:
  figures/lstsq_projection.png   the projection geometry (3 points, 3-D)
  figures/lstsq_gd_contour.png   the GD path on the loss contour + lr sweep
  run_least_squares.json

Run:  uv run scripts/least_squares.py
"""

import json
import pathlib

import _bootstrap  # noqa: F401  (sets sys.path + FIGDIR)
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perceptron import lstsq
from perceptron.data import SEED, design_matrix, regression_1d

OUT = pathlib.Path(__file__).resolve().parents[1] / "run_least_squares.json"
LR = 0.05
STEPS = 400


def main() -> None:
    x, y = regression_1d(n=40, seed=SEED)
    X = design_matrix(x)

    w_star = lstsq.normal_equations(X, y)
    geo = lstsq.projection(X, y)
    w_gd, path, losses = lstsq.gradient_descent(X, y, lr=LR, steps=STEPS)
    edge = lstsq.stability_edge(X)
    gap = float(np.max(np.abs(w_gd - w_star)))

    print(f"data          n={len(x)}  seed={SEED}  true line y = 1.7x − 0.8 + ε(σ=0.55)")
    print(f"normal eqs    intercept {w_star[0]:+.6f}   slope {w_star[1]:+.6f}")
    print(f"GD lr={LR} × {STEPS} steps")
    print(f"              intercept {w_gd[0]:+.6f}   slope {w_gd[1]:+.6f}")
    print(f"max |Δw|      {gap:.3e}")
    print(f"loss          normal eqs {lstsq.loss(X, y, w_star):.8f}   "
          f"GD {losses[-1]:.8f}")
    print(f"orthogonality max |Xᵀr| = {geo['max_abs_orthogonality']:.3e}  "
          "(machine zero: the residual is perpendicular to every column)")
    print(f"stability     GD diverges above lr = 2/λ_max = {edge:.4f}")
    assert gap < 1e-6, gap
    assert geo["max_abs_orthogonality"] < 1e-12

    # how many steps to get within 1e-6 of the exact answer
    dists = np.max(np.abs(path - w_star), axis=1)
    reached = int(np.argmax(dists < 1e-6)) if np.any(dists < 1e-6) else -1
    print(f"GD reaches |Δw| < 1e-6 after {reached} steps "
          "(the normal equations get there in one solve)")

    # ---- figure 1: projection geometry, in 3-D with three data points -----
    # A 3-point problem is the largest one you can actually see: y lives in R³,
    # col(X) is a plane, ŷ is the shadow of y on that plane.
    x3 = np.array([-1.0, 0.4, 1.6])
    y3 = np.array([-1.9, 0.6, 1.1])
    X3 = design_matrix(x3)
    g3 = lstsq.projection(X3, y3)
    yhat3, r3 = g3["yhat"], g3["residual"]

    fig = plt.figure(figsize=(11.2, 4.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    c1, c2 = X3[:, 0], X3[:, 1]
    # the plane spanned by the two columns, kept small enough that the vectors
    # on it stay readable
    s = np.linspace(-1.3, 1.3, 12)
    S, T = np.meshgrid(s, s)
    P = S[..., None] * c1 + T[..., None] * c2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.20,
                    color="#1f918d", linewidth=0, antialiased=True)
    for vec, col, lab in ((c1, "#3fa34d", "column 1 = 1"),
                          (c2, "#a86b0e", "column 2 = x")):
        ax.quiver(0, 0, 0, *vec, color=col, lw=2, arrow_length_ratio=0.12,
                  label=lab)
    ax.quiver(0, 0, 0, *y3, color="#453781", lw=2.6, arrow_length_ratio=0.11,
              label="y (the data)")
    ax.quiver(0, 0, 0, *yhat3, color="#1f918d", lw=2.6, arrow_length_ratio=0.11,
              label="ŷ = Xw* (the fit)")
    ax.plot(*np.array([yhat3, y3]).T, color="#c1121f", lw=2.4, ls="--",
            marker="o", ms=3.5, label="r = y − ŷ  ⟂  the plane")
    ax.text(*(yhat3 + r3 / 2 + np.array([0.08, 0.08, 0.12])),
            f"‖r‖ = {np.linalg.norm(r3):.2f}", color="#c1121f", fontsize=8.5)
    lim = 2.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("y ∈ ℝ³, col(X) is a plane\nleast squares drops a perpendicular",
                 fontsize=10)
    ax.set_xlabel("obs 1")
    ax.set_ylabel("obs 2")
    ax.set_zlabel("obs 3")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(-0.14, 1.02))
    # look along the plane, roughly perpendicular to the residual, so that the
    # right angle between r and col(X) is actually visible
    ax.view_init(elev=26, azim=-4)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x, y, s=22, color="#453781", label="the 40 noisy points", zorder=3)
    xs = np.linspace(-3.2, 3.2, 20)
    ax2.plot(xs, w_star[0] + w_star[1] * xs, color="#1f918d", lw=2.2,
             label=f"fit  y = {w_star[1]:.3f}x {w_star[0]:+.3f}")
    for xi, yi in zip(x, y):
        ax2.plot([xi, xi], [yi, w_star[0] + w_star[1] * xi],
                 color="#c1121f", lw=0.8, alpha=0.55)
    ax2.set_title(f"the residuals it minimises  (RSS = {geo['rss']:.3f},\n"
                  f"max |Xᵀr| = {geo['max_abs_orthogonality']:.1e})", fontsize=10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "lstsq_projection.png", dpi=110)
    plt.close(fig)

    # ---- figure 2: GD on the loss contour, plus a learning-rate sweep -----
    b_lo, b_hi = w_star[0] - 2.2, w_star[0] + 2.2
    m_lo, m_hi = w_star[1] - 2.2, w_star[1] + 2.2
    B, M = np.meshgrid(np.linspace(b_lo, b_hi, 220), np.linspace(m_lo, m_hi, 220))
    L = np.empty_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            L[i, j] = lstsq.loss(X, y, np.array([B[i, j], M[i, j]]))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    cs = ax.contourf(B, M, L, levels=28, cmap="viridis")
    ax.contour(B, M, L, levels=14, colors="white", linewidths=0.35, alpha=0.5)
    fig.colorbar(cs, ax=ax, label="mean squared error")
    ax.plot(path[:, 0], path[:, 1], "-o", color="#c1121f", ms=2.4, lw=1.3,
            label=f"GD path, lr = {LR}")
    ax.scatter([w_star[0]], [w_star[1]], marker="*", s=190, color="white",
               edgecolors="#c1121f", zorder=5, label="normal-equations solution")
    ax.set_xlabel("intercept w₀")
    ax.set_ylabel("slope w₁")
    ax.set_title("the loss is a bowl; GD walks to the one point\n"
                 "the normal equations name outright", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    lstar = lstsq.loss(X, y, w_star)
    for lr, col in ((0.01, "#453781"), (LR, "#1f918d"), (0.30, "#a86b0e"),
                    (0.55, "#c1121f")):
        _, _, ls = lstsq.gradient_descent(X, y, lr=lr, steps=STEPS)
        # plot the *excess* loss: it decays geometrically, so log scale turns
        # each learning rate into a straight line whose slope is its rate
        ex = np.maximum(np.nan_to_num(ls - lstar, nan=np.inf), 1e-17)
        CEIL = 1e6
        keep = int(np.argmax(ex > CEIL)) if np.any(ex > CEIL) else len(ex)
        label = f"lr = {lr}"
        if lr > edge:
            label += "  (past the edge)"
        ax.plot(np.arange(keep), ex[:keep], color=col, lw=1.8, label=label)
    ax.axhline(1e-16, color="#5c6b67", ls="--", lw=1.2,
               label="double-precision floor")
    ax.set_yscale("log")
    ax.set_xlim(0, 260)
    ax.set_ylim(1e-17, 1e6)
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel("excess loss  L − L*  (log)")
    ax.set_title(f"stability edge 2/λ_max = {edge:.3f}: above it,\n"
                 "the same code diverges", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "lstsq_gd_contour.png", dpi=110)
    plt.close(fig)

    blob = {
        "seed": SEED, "n": len(x), "lr": LR, "steps": STEPS,
        "w_normal": [float(v) for v in w_star],
        "w_gd": [float(v) for v in w_gd],
        "max_abs_diff": gap,
        "loss_normal": lstsq.loss(X, y, w_star),
        "loss_gd": float(losses[-1]),
        "rss": geo["rss"],
        "max_abs_orthogonality": geo["max_abs_orthogonality"],
        "stability_edge": edge,
        "steps_to_1e-6": reached,
        "loss_curve_lr005": [float(v) for v in losses[:120]],
        "path_lr005": [[float(a), float(b)] for a, b in path[:120]],
        "points": [[float(a), float(b)] for a, b in zip(x, y)],
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"wrote {FIGDIR / 'lstsq_projection.png'}")
    print(f"wrote {FIGDIR / 'lstsq_gd_contour.png'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
