"""How the margin controls the work: mistakes vs (R/γ)² across many datasets.

Sweeps the geometric margin used to build the data, retrains from scratch at
each setting, and plots the empirical mistake count against Novikoff's bound on
log axes. The bound is not tight, and it is not even the right *shape* for this
data: fitting both curves' log-log slopes gives γ^-2.11 for the bound against
γ^-1.00 for the mistakes actually made. A worst-case guarantee is a promise
about the worst case, not a prediction about yours.

Produces:
  figures/margin_sweep.png
  run_margin_sweep.json

Run:  uv run scripts/margin_sweep.py
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
from perceptron.data import SEED, augment, design_matrix, separable_2d
from perceptron.lstsq import normal_equations

MARGINS = [0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.55, 0.8, 1.1]
SEEDS = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]
N = 200
OUT = pathlib.Path(__file__).resolve().parents[1] / "run_margin_sweep.json"


def main() -> None:
    rows = []
    print(f"{'margin':>7} {'γ (mean)':>10} {'bound':>12} {'mistakes':>20} {'ratio':>8}")
    for m in MARGINS:
        mis, bounds, gammas = [], [], []
        for s in SEEDS:
            X, y = separable_2d(n=N, margin=m, seed=s)
            Xa = augment(X)
            run = rosenblatt.train(Xa, y, max_epochs=4000)
            info = rosenblatt.novikoff_bound(Xa, y)
            assert run.converged, f"did not converge at margin {m}, seed {s}"
            assert run.mistakes <= info["bound"], (m, s, run.mistakes, info["bound"])
            mis.append(run.mistakes)
            bounds.append(info["bound"])
            gammas.append(info["gamma"])
        row = {
            "margin": m,
            "gamma_mean": float(np.mean(gammas)),
            "bound_mean": float(np.mean(bounds)),
            "mistakes_mean": float(np.mean(mis)),
            "mistakes_min": int(np.min(mis)),
            "mistakes_max": int(np.max(mis)),
            "ratio": float(np.mean(mis) / np.mean(bounds)),
        }
        rows.append(row)
        print(f"{m:7.2f} {row['gamma_mean']:10.4f} {row['bound_mean']:12.1f} "
              f"{row['mistakes_mean']:9.1f} [{row['mistakes_min']:4d},"
              f"{row['mistakes_max']:5d}] {row['ratio']:8.4f}")

    # x axis is the *realised* augmented margin, not the requested one: at small
    # requested margins the nearest of 200 sampled points usually lands further
    # out than the rejection threshold, so the two differ.
    xs = np.array([r["gamma_mean"] for r in rows])
    bound = np.array([r["bound_mean"] for r in rows])
    emp = np.array([r["mistakes_mean"] for r in rows])
    lo = np.array([r["mistakes_min"] for r in rows])
    hi = np.array([r["mistakes_max"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(xs, bound, "o--", color="#c1121f", lw=2,
            label="Novikoff bound  (R/γ)²")
    ax.plot(xs, emp, "o-", color="#1f918d", lw=2,
            label=f"mistakes actually made (mean of {len(SEEDS)} seeds)")
    ax.fill_between(xs, lo, hi, color="#1f918d", alpha=0.16,
                    label="min–max across seeds")
    # fit the log-log slopes with the module's own normal equations
    A = design_matrix(np.log(xs))
    p_bound = normal_equations(A, np.log(bound))[1]
    p_emp = normal_equations(A, np.log(emp))[1]
    print(f"\nlog-log slope   bound γ^{p_bound:.2f}   empirical γ^{p_emp:.2f}")

    ref = emp[-1] * (xs[-1] / xs) ** 2
    ax.plot(xs, ref, ":", color="#5c6b67", lw=1.4,
            label="a pure 1/γ² reference, anchored on the right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("realised margin γ of the data (augmented space)")
    ax.set_ylabel("mistakes before convergence")
    ax.set_title(f"The bound really is 1/γ² (fitted slope {p_bound:.2f}); this data\n"
                 f"only costs about 1/γ (fitted slope {p_emp:.2f})")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "margin_sweep.png", dpi=110)
    plt.close(fig)

    OUT.write_text(json.dumps({"n": N, "seeds": SEEDS, "rows": rows,
                               "loglog_slope_bound": float(p_bound),
                               "loglog_slope_empirical": float(p_emp)},
                              indent=2) + "\n")
    print(f"\nratio range {min(r['ratio'] for r in rows):.4f} … "
          f"{max(r['ratio'] for r in rows):.4f}")
    print(f"wrote {FIGDIR / 'margin_sweep.png'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
