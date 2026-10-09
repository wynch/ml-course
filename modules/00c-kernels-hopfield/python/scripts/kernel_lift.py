"""The kernel trick, made literal.

Two figures:

figures/kernel_lift_3d.png
    left   — concentric rings in 2D, where no straight line can win;
    middle — the same points after the *explicit* quadratic map
             φ(x) = (x₁², √2·x₁x₂, x₂²), with the plane the dual SVM found;
    right  — the picture-book paraboloid lift (x₁, x₂, ‖x‖²) for comparison.

figures/kernel_boundaries.png
    the same rings fitted with a linear, a quadratic and an RBF kernel — same
    solver, one line changed — with the support vectors ringed.

Run:  uv run scripts/kernel_lift.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernelmem.lift import check_identity, phi_paraboloid, phi_quadratic
from kernelmem.svm import DualSVM, circles, kernel_matrix

IN = "#5ec962"    # inner ring, class +1
OUT = "#453781"   # outer ring, class −1


def main() -> None:
    X, y = circles()
    err = check_identity(X)
    print(f"max |<phi(a),phi(b)> - (a.b)^2| over all pairs = {err:.3e}")

    # the SAME dual solver, on the quadratic kernel
    poly = DualSVM(kernel="poly", params={"degree": 2}, iters=6000).fit(X, y)
    acc = float((poly.predict(X) == y).mean())
    sv = poly.support_mask()
    print(f"quadratic kernel: train accuracy {acc:.3f}, "
          f"{int(sv.sum())} support vectors, margin {poly.margin():.4f}")

    # In the explicit feature space the same solution is an ordinary hyperplane,
    # w = Σ αᵢ yᵢ φ(xᵢ). Computing it is only possible because φ is small here.
    F = phi_quadratic(X)
    w3 = (poly.alpha * y) @ F
    b3 = poly.b
    print(f"feature-space plane: w = ({w3[0]:+.4f}, {w3[1]:+.4f}, {w3[2]:+.4f})  b = {b3:+.4f}")
    print("  pulled back to 2D that plane reads "
          f"{w3[0]:+.4f}·x1² {np.sqrt(2) * w3[1]:+.4f}·x1x2 {w3[2]:+.4f}·x2² {b3:+.4f} = 0")
    print(f"  → an ellipse with semi-axes ≈ {np.sqrt(-b3 / w3[0]):.4f} and "
          f"{np.sqrt(-b3 / w3[2]):.4f}: a near-circle, not exactly one, because the "
          "sampled rings are not exactly symmetric")

    # the max-margin threshold in the 1D feature ‖x‖², used for the paraboloid panel
    r2_in = (X[y == 1] ** 2).sum(1).max()
    r2_out = (X[y == -1] ** 2).sum(1).min()
    r2 = 0.5 * (r2_in + r2_out)
    print(f"  paraboloid view: inner radii² top out at {r2_in:.4f}, outer start at "
          f"{r2_out:.4f} → cut at ‖x‖² = {r2:.4f}")

    # sanity: the explicit-φ hyperplane and the kernel decision function agree
    explicit = F @ w3 + b3
    kernelised = poly.decision_function(X)
    print(f"max |explicit - kernelised| decision value = "
          f"{np.abs(explicit - kernelised).max():.3e}")

    # ---- figure 1: the lift ----------------------------------------------
    fig = plt.figure(figsize=(13.6, 5.2))

    ax0 = fig.add_subplot(1, 3, 1)
    for cls, colour, name in ((1, IN, "class +1 (inner)"), (-1, OUT, "class −1 (outer)")):
        m = y == cls
        ax0.scatter(X[m, 0], X[m, 1], s=26, c=colour, edgecolors="white",
                    linewidths=0.5, label=name)
    ax0.set_aspect("equal")
    ax0.set_xlabel("$x_1$")
    ax0.set_ylabel("$x_2$")
    ax0.set_title("2D · no line can do this", pad=14)
    ax0.legend(fontsize=8, loc="upper right")

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    for cls, colour in ((1, IN), (-1, OUT)):
        m = y == cls
        ax1.scatter(F[m, 0], F[m, 1], F[m, 2], s=18, c=colour, depthshade=False)
    g = np.linspace(0, max(F[:, 0].max(), F[:, 2].max()) * 1.05, 12)
    G1, G2 = np.meshgrid(g, np.linspace(F[:, 1].min(), F[:, 1].max(), 12))
    G3 = -(w3[0] * G1 + w3[1] * G2 + b3) / w3[2]
    ax1.plot_surface(G1, G2, G3, color="#1f918d", alpha=0.30, linewidth=0)
    ax1.set_xlabel("$z_1 = x_1^2$", labelpad=-2)
    ax1.set_ylabel(r"$z_2 = \sqrt{2}\,x_1x_2$", labelpad=-2)
    ax1.set_zlabel("$z_3 = x_2^2$", labelpad=2)
    ax1.tick_params(labelsize=7, pad=-1)
    ax1.set_title(r"$\varphi(x)=(x_1^2,\sqrt{2}x_1x_2,x_2^2)$" "\nthe real feature space of $(a^\\top b)^2$",
                  fontsize=10, pad=14)
    ax1.view_init(elev=18, azim=-58)

    P = phi_paraboloid(X)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    for cls, colour in ((1, IN), (-1, OUT)):
        m = y == cls
        ax2.scatter(P[m, 0], P[m, 1], P[m, 2], s=18, c=colour, depthshade=False)
    gg = np.linspace(-2.6, 2.6, 24)
    Q1, Q2 = np.meshgrid(gg, gg)
    ax2.plot_surface(Q1, Q2, np.full_like(Q1, r2), color="#1f918d", alpha=0.30, linewidth=0)
    ax2.set_xlabel("$x_1$", labelpad=-4)
    ax2.set_ylabel("$x_2$", labelpad=-4)
    ax2.set_zlabel(r"$\|x\|^2$", labelpad=2)
    ax2.tick_params(labelsize=7, pad=-1)
    ax2.set_title(r"$\varphi(x)=(x_1,x_2,\|x\|^2)$" "\nthe picture-book lift (a different kernel)",
                  fontsize=10, pad=14)
    ax2.view_init(elev=16, azim=-62)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.10, wspace=0.14)
    out = FIGDIR / "kernel_lift_3d.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- figure 2: three kernels, one solver ------------------------------
    # The rings are not linearly separable, so the *hard*-margin dual is
    # unbounded for the linear kernel — there is no finite optimum to find.
    # The linear panel therefore gets a box constraint C = 1 (the soft margin);
    # the two curved kernels stay hard-margin because they separate cleanly.
    setups = [
        ("linear", {}, 1.0, "linear · $K=a^{\\top}b$ (soft, $C=1$)"),
        ("poly", {"degree": 2}, None, "quadratic · $K=(a^{\\top}b)^2$"),
        ("rbf", {"gamma": 0.5}, None, "RBF · $K=e^{-0.5\\|a-b\\|^2}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    pad = 0.7
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 260)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 260)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])

    for ax, (kind, params, C, title) in zip(axes, setups):
        m = DualSVM(kernel=kind, params=params, C=C, iters=6000).fit(X, y)
        Z = m.decision_function(grid).reshape(GX.shape)
        a = float((m.predict(X) == y).mean())
        s = m.support_mask()
        ax.contourf(GX, GY, Z, levels=[-1e9, 0, 1e9], colors=["#453781", "#5ec962"], alpha=0.13)
        ax.contour(GX, GY, Z, levels=[0], colors="#1c2422", linewidths=1.6)
        ax.contour(GX, GY, Z, levels=[-1, 1], colors="#1c2422", linewidths=0.8,
                   linestyles="dashed")
        for cls, colour in ((1, IN), (-1, OUT)):
            mm = y == cls
            ax.scatter(X[mm, 0], X[mm, 1], s=22, c=colour, edgecolors="white", linewidths=0.5)
        ax.scatter(X[s, 0], X[s, 1], s=130, facecolors="none", edgecolors="#c1121f",
                   linewidths=1.4)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\ntrain acc {a * 100:.1f}% · {int(s.sum())} SVs", fontsize=10)
        ax.set_xlabel("$x_1$")
        print(f"{kind:7s} acc {a:.4f}  SVs {int(s.sum()):3d}  margin {m.margin():.4f}")
    axes[0].set_ylabel("$x_2$")

    fig.tight_layout()
    out = FIGDIR / "kernel_boundaries.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # the Gram matrix identity, printed for the README
    K_direct = kernel_matrix(X, X, "poly", degree=2)
    K_via_phi = F @ F.T
    print(f"Gram via kernel vs via phi: max abs diff {np.abs(K_direct - K_via_phi).max():.3e}")


if __name__ == "__main__":
    main()
