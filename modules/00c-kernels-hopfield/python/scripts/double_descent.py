"""Bias–variance, then the thing that broke it: double descent.

Produces:
  figures/bias_variance.png  — the classical decomposition in the
      underparameterised regime, plus three fits (underfit / right / overfit).
  figures/double_descent.png — the full sweep: the U, the spike exactly at
      p = n, and the second descent, on one fixed dataset and as a median over
      200 fresh datasets. A right-hand panel carries the sweep out to degree 80
      and plots ‖β‖, which is what actually explodes.

Run:  uv run scripts/double_descent.py   (~40 s)
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernelmem import polyreg as pr

DEGREES = list(range(1, 31))
LONG = list(range(1, 81))
CLASSICAL = list(range(1, 13))
TRIALS = 200


def main() -> None:
    x, y, xt, yt = pr.make_dataset()
    n = len(x)
    print(f"training set: n = {n} points, sigma = {pr.SIGMA}, seed = {pr.DATA_SEED}")
    print(f"interpolation threshold at p = degree + 1 = n = {n}  →  degree {n - 1}")

    single = pr.sweep(x, y, xt, yt, LONG)
    med = pr.sweep_averaged(LONG, trials=TRIALS, seed=0)

    peak_i = int(np.argmax(single["test"]))
    peak_med = int(np.argmax(med["test"]))
    print(f"single dataset : test error peaks at degree {LONG[peak_i]} "
          f"(p = {LONG[peak_i] + 1}) at MSE {single['test'][peak_i]:.4e}")
    print(f"median of {TRIALS}: test error peaks at degree {LONG[peak_med]} "
          f"(p = {LONG[peak_med] + 1}) at MSE {med['test'][peak_med]:.4e}")

    pre = single["test"][: n - 2]
    best_pre = int(np.argmin(pre))
    post = single["test"][n:]
    best_post = int(np.argmin(post))
    print(f"classical sweet spot: degree {LONG[best_pre]}, test MSE {pre[best_pre]:.5f}")
    print(f"second-descent floor: degree {LONG[n + best_post]}, "
          f"test MSE {post[best_post]:.5f}")
    print(f"degree 30 test MSE {single['test'][29]:.5f} · "
          f"degree 80 test MSE {single['test'][79]:.5f}")
    print(f"train MSE at degree {n - 1}: {single['train'][n - 2]:.3e} "
          f"(interpolation starts here and never stops)")

    # ---- bias / variance ---------------------------------------------------
    bv = pr.bias_variance(CLASSICAL, trials=300, seed=5)
    tot = np.array(bv["bias2"]) + np.array(bv["variance"])
    best = int(np.argmin(tot))
    print(f"bias-variance: total error minimised at degree {CLASSICAL[best]} "
          f"(bias² {bv['bias2'][best]:.4f} + variance {bv['variance'][best]:.4f} "
          f"= {tot[best]:.4f})")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    axa.plot(CLASSICAL, bv["bias2"], "o-", color="#453781", lw=1.8, label="bias²")
    axa.plot(CLASSICAL, bv["variance"], "s-", color="#c1121f", lw=1.8, label="variance")
    axa.plot(CLASSICAL, tot, "^-", color="#1f918d", lw=2.0, label="bias² + variance")
    axa.axvline(CLASSICAL[best], color="#5c6b67", ls=":", lw=1.2)
    axa.annotate(f"sweet spot\ndegree {CLASSICAL[best]}", (CLASSICAL[best], tot[best]),
                 xytext=(14, 26), textcoords="offset points", fontsize=9, color="#5c6b67")
    axa.set_yscale("log")
    axa.set_xlabel("polynomial degree")
    axa.set_ylabel("error on the test grid")
    axa.set_title(f"The classical trade-off (300 datasets of n = {n})")
    axa.legend(fontsize=9)
    axa.grid(alpha=0.25)
    # honesty note: bias² is estimated as (mean of 300 fits − truth)²; once the
    # variance is in the thousands that sample mean is itself noisy, so the
    # bias² curve turning back up past degree ~7 is Monte-Carlo error, not bias.
    axa.text(0.03, 0.97,
             "bias² is a 300-sample estimate — past degree ≈7\n"
             "the variance swamps it and it stops being reliable",
             transform=axa.transAxes, va="top", fontsize=7, color="#5c6b67")

    show = [1, CLASSICAL[best], 12]
    colours = ["#c1121f", "#1f918d", "#453781"]
    grid = np.linspace(-1, 1, 400)
    axb.plot(grid, pr.truth(grid), color="#5c6b67", lw=2.2, label="truth  sin(3x)+0.35x")
    axb.scatter(x, y, s=34, c="#1c2422", zorder=4, label=f"{n} noisy samples")
    for d, c in zip(show, colours):
        beta = pr.fit_min_norm(x, y, d)
        axb.plot(grid, pr.predict(beta, grid, d), color=c, lw=1.6, label=f"degree {d}")
    axb.set_ylim(-1.9, 1.9)
    axb.set_xlabel("x")
    axb.set_ylabel("y")
    axb.set_title("Underfit, fit, overfit — the same 20 points")
    axb.legend(fontsize=8, loc="lower right")
    axb.grid(alpha=0.25)

    fig.tight_layout()
    out = FIGDIR / "bias_variance.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- double descent ----------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})

    d30 = DEGREES
    ax.plot(d30, single["test"][:30], "o-", color="#1f918d", lw=1.9, ms=4,
            label="test MSE · one fixed dataset")
    ax.plot(d30, med["test"][:30], "-", color="#453781", lw=1.6,
            label=f"test MSE · median of {TRIALS} datasets")
    # the train MSE is exactly zero past the threshold; clamp it so the log axis
    # does not stretch to 1e-30 and flatten everything else
    ax.plot(d30, np.maximum(single["train"][:30], 1e-12), "-", color="#c1121f", lw=1.4,
            label="train MSE · one dataset (clamped at 1e−12)")
    ax.axvline(n - 1, color="#5c6b67", ls="--", lw=1.4)
    ax.annotate(f"interpolation threshold\np = n = {n}  (degree {n - 1})",
                (n - 1, 1e-7), xytext=(9, 0), textcoords="offset points",
                fontsize=9, color="#5c6b67")
    ax.set_yscale("log")
    ax.set_ylim(3e-14, 1e15)
    ax.set_xlabel("polynomial degree  (parameters p = degree + 1)")
    ax.set_ylabel("mean squared error")
    ax.set_title("Double descent: the U, the spike, the second descent")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ins = ax.inset_axes([0.09, 0.07, 0.36, 0.26])
    ins.plot(d30[:12], med["test"][:12], "-", color="#453781", lw=1.5)
    ins.plot(d30[:12], single["test"][:12], "o-", color="#1f918d", lw=1.3, ms=3)
    ins.set_yscale("log")
    ins.set_title("the classical U, zoomed", fontsize=7)
    ins.tick_params(labelsize=6)

    ax2.plot(LONG, single["test"], "-", color="#1f918d", lw=1.7, label="test MSE")
    ax2.plot(LONG, med["test"], "-", color="#453781", lw=1.4,
             label=f"test MSE, median of {TRIALS}")
    ax2.axvline(n - 1, color="#5c6b67", ls="--", lw=1.2)
    ax2.axhline(pre[best_pre], color="#c1121f", ls=":", lw=1.4)
    ax2.annotate(f"classical best, {pre[best_pre]:.4f}", (44, pre[best_pre]),
                 xytext=(0, 6), textcoords="offset points", ha="left",
                 fontsize=8, color="#c1121f")
    ax2.set_yscale("log")
    ax2.set_xlabel("polynomial degree, carried out to 80")
    ax2.set_ylabel("test MSE")
    ax2.set_title("Honest ending: it comes back down, not all the way")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25)

    ins2 = ax2.inset_axes([0.55, 0.44, 0.42, 0.28])
    ins2.plot(LONG, single["beta_norm"], color="#c1121f", lw=1.4)
    ins2.axvline(n - 1, color="#5c6b67", ls="--", lw=1.0)
    ins2.set_yscale("log")
    ins2.set_title(r"$\|\beta\|_2$ — what actually blows up", fontsize=7)
    ins2.tick_params(labelsize=6)

    fig.tight_layout()
    out = FIGDIR / "double_descent.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    print(f"||beta|| at degree {n - 1}: {single['beta_norm'][n - 2]:.4e}   "
          f"at degree 30: {single['beta_norm'][29]:.4f}   "
          f"at degree 80: {single['beta_norm'][79]:.4f}")


if __name__ == "__main__":
    main()
