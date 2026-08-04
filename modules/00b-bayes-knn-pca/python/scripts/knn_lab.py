"""k-NN from scratch, and the Cover-Hart bound measured rather than asserted.

The data is two equal-prior spherical Gaussians separated by ``sep = 2`` with
``sigma = 1``, so:

    Bayes error  R*        = Phi(-sep / (2 sigma)) = Phi(-1) = 0.158655
    eta(x) = P(y=1 | x)    = sigmoid(sep * x0 / sigma^2)
    1-NN limit             = E[2 eta (1 - eta)]        (integrated numerically)
    Cover-Hart ceiling     = 2 R* (1 - R*)

Then we run 1-NN on training sets from 50 to 50,000 points and watch the
measured error walk down towards that limit — always above R*, always below the
ceiling.

Produces: figures/knn_regions.png, figures/knn_accuracy_vs_k.png,
          figures/cover_hart.png

Run:  uv run scripts/knn_lab.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import json
import time

import numpy as np

from origins.data import two_gaussians
from origins.knn import (
    KNN,
    asymptotic_1nn_error,
    bayes_error_two_gaussians,
    cover_hart_bound,
    eta_two_gaussians,
)
from origins.plots import ACCENT, MUTED, T0, WARN, finish, plt

SEP = 2.0
SIGMA = 1.0
N_TEST = 20_000
K_GRID = [1, 3, 5, 9, 15, 25, 41, 65, 101, 151, 201, 301]
N_GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
REPEATS = 3
PANEL_KS = [1, 5, 25, 101]


def theory() -> dict:
    """Everything computable without touching a classifier."""
    R_star = bayes_error_two_gaussians(SEP, SIGMA)
    bound = cover_hart_bound(R_star, n_classes=2)
    # Monte-Carlo the two expectations over the true marginal of x
    Xbig, _ = two_gaussians(2_000_000, sep=SEP, sigma=SIGMA, seed=99)
    eta = eta_two_gaussians(Xbig, SEP, SIGMA)
    R_star_mc = float(np.minimum(eta, 1 - eta).mean())
    R_1nn = asymptotic_1nn_error(eta)
    print("Theory for two spherical Gaussians, sep=2.0, sigma=1.0")
    print(f"  Bayes error R*                     = {R_star:.6f}   (= Phi(-1))")
    print(f"  R* by Monte Carlo, n=2e6           = {R_star_mc:.6f}   "
          f"(diff {abs(R_star - R_star_mc):.2e})")
    print(f"  asymptotic 1-NN error E[2eta(1-eta)] = {R_1nn:.6f}   "
          f"= {R_1nn / R_star:.4f} x R*")
    print(f"  Cover-Hart ceiling 2R*(1-R*)       = {bound:.6f}   "
          f"= {bound / R_star:.4f} x R*")
    assert R_star <= R_1nn <= bound, (R_star, R_1nn, bound)
    print("  check: R* <= asymptotic 1-NN <= ceiling  ✓")
    return {"R_star": R_star, "R_star_mc": R_star_mc, "R_1nn": R_1nn, "bound": bound}


def fig_regions(R_star: float) -> None:
    Xtr, ytr = two_gaussians(600, sep=SEP, sigma=SIGMA, seed=1)
    Xte, yte = two_gaussians(N_TEST, sep=SEP, sigma=SIGMA, seed=2)
    lo, hi = -4.6, 4.6
    g = np.linspace(lo, hi, 260)
    G0, G1 = np.meshgrid(g, g)
    P = np.column_stack([G0.ravel(), G1.ravel()])

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4))
    for ax, k in zip(axes, PANEL_KS):
        model = KNN(k=k).fit(Xtr, ytr)
        Z = model.predict(P).reshape(G0.shape)
        err = 1.0 - model.score(Xte, yte)
        ax.contourf(G0, G1, Z, levels=[-0.5, 0.5, 1.5], colors=["#f0e3c8", "#d9ecea"])
        ax.contour(G0, G1, Z, levels=[0.5], colors=[WARN], linewidths=1.3)
        ax.axvline(0.0, color="k", lw=1.4, ls="--")
        for c, col, mark in ((0, T0, "o"), (1, ACCENT, "^")):
            m = ytr == c
            ax.scatter(Xtr[m, 0], Xtr[m, 1], s=7, c=col, marker=mark, lw=0, alpha=0.7)
        ax.set_title(f"k = {k}\ntest error {err:.4f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("x0")
        if ax is axes[0]:
            ax.set_ylabel("x1")
    fig.suptitle(
        f"k-NN decision regions, 600 training points — dashed line is the Bayes-optimal "
        f"boundary x0 = 0 (error {R_star:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, FIGDIR / "knn_regions.png")


def fig_accuracy_vs_k(th: dict) -> dict:
    Xtr, ytr = two_gaussians(1000, sep=SEP, sigma=SIGMA, seed=1)
    Xte, yte = two_gaussians(N_TEST, sep=SEP, sigma=SIGMA, seed=2)
    rows = []
    for k in K_GRID:
        m = KNN(k=k).fit(Xtr, ytr)
        rows.append((k, 1.0 - m.score(Xtr, ytr), 1.0 - m.score(Xte, yte)))
    print("\nError vs k (1,000 train / 20,000 test)")
    print("     k   train err   test err")
    for k, tr, te in rows:
        print(f"  {k:4d}   {tr:9.4f}   {te:8.4f}")
    best = min(rows, key=lambda r: r[2])
    print(f"  best k = {best[0]} at test error {best[2]:.4f} "
          f"({best[2] / th['R_star']:.3f} x R*)")

    ks = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogx(ks, [r[1] for r in rows], "o-", color=MUTED, lw=1.4, ms=4, label="train error")
    ax.semilogx(ks, [r[2] for r in rows], "o-", color=T0, lw=2.0, ms=5, label="test error")
    ax.axhline(th["R_star"], color="k", ls="--", lw=1.2,
               label=f"Bayes error R* = {th['R_star']:.4f}")
    ax.axhline(th["R_1nn"], color=WARN, ls=":", lw=1.4,
               label=f"1-NN limit E[2η(1−η)] = {th['R_1nn']:.4f}")
    ax.plot([best[0]], [best[2]], "*", color=ACCENT, ms=14, zorder=5)
    ax.annotate(f"best k = {best[0]}", xy=(best[0], best[2]), xytext=(best[0] * 1.3, best[2] + 0.03),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax.set_xlabel("k  (log scale)")
    ax.set_ylabel("error rate")
    ax.set_title("k-NN error vs k — 1,000 training points, 20,000 test points")
    ax.set_ylim(0, 0.34)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    finish(fig, FIGDIR / "knn_accuracy_vs_k.png")
    return {"rows_k": rows, "best_k": best[0], "best_test_err": best[2]}


def fig_cover_hart(th: dict) -> dict:
    """1-NN and sqrt(n)-NN error as the training set grows."""
    Xte, yte = two_gaussians(N_TEST, sep=SEP, sigma=SIGMA, seed=2)
    err1, errk, ks_used = [], [], []
    t0 = time.time()
    for n in N_GRID:
        e1, ek = [], []
        for r in range(REPEATS):
            Xtr, ytr = two_gaussians(n, sep=SEP, sigma=SIGMA, seed=1000 + r)
            e1.append(1.0 - KNN(k=1).fit(Xtr, ytr).score(Xte, yte))
            k = max(1, int(round(np.sqrt(n))) | 1)   # odd k ~ sqrt(n)
            ek.append(1.0 - KNN(k=k).fit(Xtr, ytr).score(Xte, yte))
        err1.append(float(np.mean(e1)))
        errk.append(float(np.mean(ek)))
        ks_used.append(max(1, int(round(np.sqrt(n))) | 1))
        print(f"  n = {n:6d}   1-NN err {err1[-1]:.4f}   "
              f"{ks_used[-1]:3d}-NN err {errk[-1]:.4f}   "
              f"({err1[-1] / th['R_star']:.3f} x R*)")
    print(f"  ({time.time() - t0:.1f}s, {REPEATS} training draws averaged per n)")

    over = [n for n, e in zip(N_GRID, err1) if e > th["bound"]]
    print(f"  1-NN error above the Cover-Hart ceiling at n = {over or 'never'}")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(th["R_star"], th["bound"], color=T0, alpha=0.08)
    ax.semilogx(N_GRID, err1, "o-", color=T0, lw=2.0, ms=5, label="measured 1-NN error")
    ax.semilogx(N_GRID, errk, "s-", color=ACCENT, lw=1.6, ms=4,
                label="measured k-NN error, k ≈ √n")
    ax.axhline(th["bound"], color=WARN, lw=1.6,
               label=f"Cover-Hart ceiling 2R*(1−R*) = {th['bound']:.4f}")
    ax.axhline(th["R_1nn"], color=WARN, ls=":", lw=1.6,
               label=f"1-NN limit E[2η(1−η)] = {th['R_1nn']:.4f}")
    ax.axhline(th["R_star"], color="k", ls="--", lw=1.4,
               label=f"Bayes error R* = {th['R_star']:.4f}")
    ax.text(58, th["bound"] - 0.012, "everything Cover & Hart allow", fontsize=8, color=T0)
    ax.set_xlabel("training-set size n  (log scale)")
    ax.set_ylabel("error on 20,000 held-out points")
    ax.set_title("Cover-Hart, measured: 1-NN pays at most twice the Bayes error")
    ax.set_ylim(0.13, 0.33)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    finish(fig, FIGDIR / "cover_hart.png")
    return {"n_grid": N_GRID, "err_1nn": err1, "err_sqrtn": errk, "ks_sqrtn": ks_used}


def main() -> None:
    th = theory()
    fig_regions(th["R_star"])
    r_k = fig_accuracy_vs_k(th)
    print("\n1-NN error as n grows (20,000 fixed test points)")
    r_n = fig_cover_hart(th)
    out = FIGDIR.parent / "python" / "knn_results.json"
    out.write_text(json.dumps({**th, **r_n, "best_k": r_k["best_k"],
                              "best_test_err": r_k["best_test_err"],
                              "rows_k": r_k["rows_k"]}, indent=1))
    print(f"\n  wrote {out.name} (numbers reused by the explorable and the quiz)")


if __name__ == "__main__":
    main()
