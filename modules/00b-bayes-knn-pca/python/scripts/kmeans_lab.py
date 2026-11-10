"""k-means: Lloyd against the moving-average rule, from the same starts.

Two data sets the module already has:

* the two spherical Gaussians of ``origins.data.two_gaussians`` (2-D, 10,000
  points), with the gap between the means widened from 2 to 3, 4, 5 and 6, and
  k = 4. The best answer puts two centres on each blob; a start with three
  centres on one blob is the trap.
* FashionMNIST, 10,000 training images projected on their top 50 principal
  components (Part 4), k = 32, scored also on 5,000 test images.

For every start, Lloyd runs 25 passes; the moving-average rule runs 10 passes
plus its exact final pass. Both draw their k starting centres from the data
with the same seed, and both use ``origins.kmeans.assign`` to find each
point's nearest centre.

The batch is fixed by hand at 256. The starting step is the value the step law
gives for that batch and each data set's n, k and T = 10 (the paper's eq. 6 and
Table 2; the four-digit coefficients of the Faiss code README, its eq. 3); the
law itself is not in the code. Dead-cluster revival is off.

Two checks on the reading of the 2-D result: the stuck shares at lines of 1%
and 5% as well as 2%, and the moving average at gap 6 with smaller batches.

Produces: figures/kmeans_lloyd_vs_ema.png, python/kmeans_results.json

Run:  uv run scripts/kmeans_lab.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import json
import time

import numpy as np

from origins.data import fashion_available, load_fashion_mnist, two_gaussians
from origins.kmeans import assign, init_from_data, kmeans_error, lloyd, moving_average_kmeans
from origins.knn import KNN
from origins.pca import PCA
from origins.plots import INK, MUTED, T0, WARN, finish, plt

N_POINTS = 10_000
K_2D = 4
GAPS = [3.0, 4.0, 5.0, 6.0]
MAIN_GAP = 5.0
STARTS_2D = 300
LLOYD_PASSES = 25
EMA_PASSES = 10
BATCH = 256
# step law (paper eq. 6; Faiss code README eq. 3): 0.8244 k^-0.4617 B^0.9003 n^-0.4498 T^-0.1057, at B = 256, T = 10
ALPHA0_2D = 0.80     # n = 10,000, k = 4   -> 0.797
ALPHA0_FASHION = 0.31  # n = 10,000, k = 32 -> 0.305
STUCK = 0.02         # "clearly worse": final error more than 2% above the best run
OTHER_LINES = [0.01, 0.05]  # the same count at other lines, to see if 2% matters
SMALL_BATCHES = [64, 32]    # gap 6, moving average only, same alpha0 and starts
K_FASHION = 32
PCA_DIMS = 50
STARTS_FASHION = 50


def check_shared_step(X: np.ndarray) -> None:
    """The two rules share the assignment; with batch = n and step 1 they coincide."""
    C0 = init_from_data(X, K_2D, seed=0)
    labels, _ = assign(X, C0)
    knn = KNN(k=1).fit(C0, np.arange(K_2D)).predict(X)
    assert np.array_equal(labels, knn), "assign() must be Part 3's 1-NN against the centres"
    _, el = lloyd(X, C0, 3)
    _, ee = moving_average_kmeans(X, C0, 3, batch=len(X), alpha0=1.0, cosine=False,
                                  finalize=False)
    assert np.allclose(el, ee, rtol=1e-10), (el, ee)
    print("  check: assign() = 1-NN against the centres; full batch + step 1 = Lloyd  ✓")


def run_2d() -> dict:
    out = {}
    for gap in GAPS:
        X, _ = two_gaussians(N_POINTS, sep=gap, sigma=1.0, seed=5)
        if gap == GAPS[0]:
            check_shared_step(X)
        t0 = time.time()
        L, E = [], []
        for s in range(STARTS_2D):
            C0 = init_from_data(X, K_2D, seed=s)
            L.append(lloyd(X, C0, LLOYD_PASSES)[1])
            E.append(moving_average_kmeans(X, C0, EMA_PASSES, batch=BATCH, alpha0=ALPHA0_2D,
                                           seed=1000 + s)[1])
        L, E = np.array(L), np.array(E)
        best = min(L[:, -1].min(), E[:, -1].min())
        row = {
            "best": float(best),
            "stuck_lloyd": float(np.mean(L[:, -1] > (1 + STUCK) * best)),
            "stuck_ema": float(np.mean(E[:, -1] > (1 + STUCK) * best)),
            "median_final_lloyd": float(np.median(L[:, -1])),
            "median_final_ema": float(np.median(E[:, -1])),
            "median_after1_lloyd": float(np.median(L[:, 1])),
            "median_after1_ema": float(np.median(E[:, 1])),
            "stuck_other_lines": {
                f"{line:.0%}": {"lloyd": float(np.mean(L[:, -1] > (1 + line) * best)),
                                "ema": float(np.mean(E[:, -1] > (1 + line) * best))}
                for line in OTHER_LINES},
            "seconds": time.time() - t0,
        }
        if gap == MAIN_GAP:
            row["curve_lloyd"] = np.quantile(L, [0.25, 0.5, 0.75], axis=0).tolist()
            row["curve_ema"] = np.quantile(E, [0.25, 0.5, 0.75], axis=0).tolist()
            # a start where Lloyd stays stuck and the moving average gets out
            pick = next(s for s in range(STARTS_2D)
                        if L[s, -1] > (1 + STUCK) * best and E[s, -1] < (1 + STUCK) * best)
            row["example_start"] = pick
        out[str(gap)] = row
        print(f"  gap {gap:.0f}: best {best:.4f}   stuck (> {STUCK:.0%} above best) "
              f"Lloyd {row['stuck_lloyd']:.1%}  moving average {row['stuck_ema']:.1%}   "
              f"median final {row['median_final_lloyd']:.4f} / {row['median_final_ema']:.4f}   "
              f"after 1 pass {row['median_after1_lloyd']:.4f} / {row['median_after1_ema']:.4f}   "
              f"({row['seconds']:.1f}s)")
        print("         other lines: " + "   ".join(
            f"{k}: Lloyd {v['lloyd']:.1%}  moving average {v['ema']:.1%}"
            for k, v in row["stuck_other_lines"].items()))
    return out


def run_gap6_small_batches(r2: dict) -> dict:
    """Does more batch noise get centres out at gap 6? Same starts, same alpha0."""
    gap = GAPS[-1]
    X, _ = two_gaussians(N_POINTS, sep=gap, sigma=1.0, seed=5)
    best = r2[str(gap)]["best"]
    out = {}
    for batch in SMALL_BATCHES:
        F = np.array([moving_average_kmeans(X, init_from_data(X, K_2D, seed=s), EMA_PASSES,
                                            batch=batch, alpha0=ALPHA0_2D, seed=1000 + s)[1][-1]
                      for s in range(STARTS_2D)])
        out[str(batch)] = {"stuck_ema": float(np.mean(F > (1 + STUCK) * best))}
        print(f"  gap {gap:.0f}, batch {batch}: moving average stuck "
              f"{out[str(batch)]['stuck_ema']:.1%}")
    return out


def run_fashion() -> dict:
    if fashion_available():
        xtr, _, xte, _ = load_fashion_mnist(train_subsample=N_POINTS, seed=0)
        source = "FashionMNIST"
    else:
        xtr, _ = two_gaussians(N_POINTS, dim=784, seed=0)
        xte, _ = two_gaussians(5_000, dim=784, seed=1)
        source = "synthetic (FashionMNIST not cached)"
    pca = PCA(PCA_DIMS, solver="eigh").fit(xtr)
    Z, Zt = pca.transform(xtr), pca.transform(xte[:5_000])
    print(f"  {source}: {len(Z):,} train / {len(Zt):,} test, projected on {PCA_DIMS} components")
    t0 = time.time()
    L, E, Lt, Et = [], [], [], []
    for s in range(STARTS_FASHION):
        C0 = init_from_data(Z, K_FASHION, seed=s)
        Cl, el = lloyd(Z, C0, LLOYD_PASSES)
        Ce, ee = moving_average_kmeans(Z, C0, EMA_PASSES, batch=BATCH, alpha0=ALPHA0_FASHION,
                                       seed=1000 + s)
        L.append(el)
        E.append(ee)
        Lt.append(kmeans_error(Zt, Cl))
        Et.append(kmeans_error(Zt, Ce))
    L, E, Lt, Et = map(np.array, (L, E, Lt, Et))
    # passes the moving average needs to drop below the same start's Lloyd@25 (train error)
    reach = [int(np.argmax(e <= l)) if (e <= l).any() else None for e, l in zip(E, L[:, -1])]
    hit = [r for r in reach if r is not None]
    res = {
        "source": source,
        "lloyd25_train_median": float(np.median(L[:, -1])),
        "ema_train_median": float(np.median(E[:, -1])),
        "lloyd25_test_median": float(np.median(Lt)),
        "ema_test_median": float(np.median(Et)),
        "after1_lloyd": float(np.median(L[:, 1])),
        "after1_ema": float(np.median(E[:, 1])),
        "share_ema_below_lloyd25": float(np.mean(E[:, -1] < L[:, -1])),
        "share_test_below": float(np.mean(Et < Lt)),
        "reach_passes": reach,
        "reach_median": float(np.median(hit)) if hit else None,
        "curve_lloyd": np.median(L, axis=0).tolist(),
        "curve_ema": np.median(E, axis=0).tolist(),
        "seconds": time.time() - t0,
    }
    print(f"  after 1 pass: Lloyd {res['after1_lloyd']:.3f}   moving average {res['after1_ema']:.3f}")
    print(f"  final train: Lloyd@25 {res['lloyd25_train_median']:.3f}   moving average "
          f"{res['ema_train_median']:.3f}   (lower on {res['share_ema_below_lloyd25']:.0%} of starts)")
    print(f"  final test:  Lloyd@25 {res['lloyd25_test_median']:.3f}   moving average "
          f"{res['ema_test_median']:.3f}   (lower on {res['share_test_below']:.0%} of starts)")
    print(f"  passes to reach the same start's Lloyd@25: median {res['reach_median']} "
          f"on the {len(hit)} of {STARTS_FASHION} starts that get there "
          f"(pass 11 is the exact pass)   ({res['seconds']:.1f}s)")
    return res


def figure(r2: dict, rf: dict) -> None:
    X, _ = two_gaussians(N_POINTS, sep=MAIN_GAP, sigma=1.0, seed=5)
    main = r2[str(MAIN_GAP)]
    s = main["example_start"]
    C0 = init_from_data(X, K_2D, seed=s)
    Cl, _ = lloyd(X, C0, LLOYD_PASSES)
    Ce, _ = moving_average_kmeans(X, C0, EMA_PASSES, batch=BATCH, alpha0=ALPHA0_2D, seed=1000 + s)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    ax = axes[0, 0]
    show = X[:: 5]
    ax.scatter(show[:, 0], show[:, 1], s=3, c="#b9c4c0", lw=0)
    ax.scatter(C0[:, 0], C0[:, 1], s=50, marker="x", c=INK, lw=1.6, label="shared start")
    ax.scatter(Cl[:, 0], Cl[:, 1], s=70, marker="o", facecolors="none", edgecolors=WARN, lw=1.8,
               label=f"Lloyd, 25 passes: {kmeans_error(X, Cl):.4f}")
    ax.scatter(Ce[:, 0], Ce[:, 1], s=60, marker="^", c=T0, lw=0,
               label=f"moving average, 10 + 1: {kmeans_error(X, Ce):.4f}")
    ax.set_title(f"(a) one start, gap {MAIN_GAP:.0f}, k = {K_2D}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[0, 1]
    for key, col, name in (("curve_lloyd", WARN, "Lloyd"), ("curve_ema", T0, "moving average")):
        q = np.array(main[key])
        xs = np.arange(q.shape[1])
        ax.fill_between(xs, q[0], q[2], color=col, alpha=0.15, lw=0)
        ax.plot(xs, q[1], "o-", color=col, ms=3, lw=1.6, label=f"{name}, median and quartiles")
    ax.set_ylim(1.30, 1.56)
    ax.set_xlabel("passes over the data (moving average: pass 11 is the exact pass)")
    ax.set_ylabel("k-means error")
    ax.set_title(f"(b) error per pass, {STARTS_2D} starts, gap {MAIN_GAP:.0f}")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    w = 0.38
    xs = np.arange(len(GAPS))
    ax.bar(xs - w / 2, [100 * r2[str(g)]["stuck_lloyd"] for g in GAPS], w, color=WARN, label="Lloyd")
    ax.bar(xs + w / 2, [100 * r2[str(g)]["stuck_ema"] for g in GAPS], w, color=T0,
           label="moving average")
    ax.set_xticks(xs, [f"{g:.0f}" for g in GAPS])
    ax.set_xlabel("gap between the two means (σ = 1)")
    ax.set_ylabel(f"% of {STARTS_2D} starts stuck > {STUCK:.0%} above best")
    ax.set_title("(c) how often each rule ends in a worse solution")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ref = rf["lloyd25_train_median"]
    for key, col, name in (("curve_lloyd", WARN, "Lloyd"), ("curve_ema", T0, "moving average")):
        c = np.array(rf[key])
        ax.plot(np.arange(len(c))[1:], 100 * (c[1:] / ref - 1), "o-", color=col, ms=3, lw=1.6,
                label=name)
    ax.axhline(0, color=MUTED, ls="--", lw=1.0, label="Lloyd after 25 passes")
    ax.set_xlabel("passes over the data")
    ax.set_ylabel("train error, % above Lloyd@25")
    ax.set_ylim(-1.0, 6.0)
    ax.set_title(f"(d) FashionMNIST, {PCA_DIMS} PCA dims, k = {K_FASHION}, median of {STARTS_FASHION}")
    ax.legend(fontsize=7)

    fig.tight_layout()
    finish(fig, FIGDIR / "kmeans_lloyd_vs_ema.png")



def main() -> None:
    t0 = time.time()
    print(f"2-D, two Gaussians, {N_POINTS:,} points, k = {K_2D}, {STARTS_2D} starts per gap")
    r2 = run_2d()
    r2_small = run_gap6_small_batches(r2)
    print(f"\nFashionMNIST projection, k = {K_FASHION}, {STARTS_FASHION} starts")
    rf = run_fashion()
    figure(r2, rf)
    out = FIGDIR.parent / "python" / "kmeans_results.json"
    out.write_text(json.dumps({
        "settings": {"n": N_POINTS, "k_2d": K_2D, "gaps": GAPS, "starts_2d": STARTS_2D,
                     "lloyd_passes": LLOYD_PASSES, "ema_passes": EMA_PASSES, "batch": BATCH,
                     "alpha0_2d": ALPHA0_2D, "alpha0_fashion": ALPHA0_FASHION,
                     "stuck": STUCK, "other_lines": OTHER_LINES,
                     "small_batches": SMALL_BATCHES, "k_fashion": K_FASHION, "pca_dims": PCA_DIMS,
                     "starts_fashion": STARTS_FASHION},
        "two_gaussians": r2, "gap6_small_batches": r2_small, "fashion": rf}, indent=1))
    print(f"\n  wrote {out.name}   ({time.time() - t0:.0f}s in all)")


if __name__ == "__main__":
    main()
