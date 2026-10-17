"""Section 3: the manifold picture.

1. PCA against intrinsic dimension: how many principal components it takes to hold 90%,
   95% and 99% of the variance, on a curved 10-d cube and on FashionMNIST pixels.
2. Straight lines leave the data: the midpoint of two test images, against the nearest
   of the 60,000 training images, compared with how far a real test image sits from its
   nearest training image. A demo, not a measurement.

    uv run python src/manifold.py --figures
"""

from __future__ import annotations

import argparse

import common
import idest
import numpy as np
import torch

N_PAIRS = 300                     # per kind: same class, different class
SHOW = 4                          # pairs drawn in the figure, per kind


def pca_spectrum(x: np.ndarray) -> np.ndarray:
    """Share of variance held by each principal component, largest first."""
    xc = x - x.mean(0)
    s = np.linalg.svd(xc, compute_uv=False)
    v = s**2
    return v / v.sum()


def components_for(share: np.ndarray, levels=(0.5, 0.9, 0.95, 0.99)) -> dict:
    c = np.cumsum(share)
    return {str(lv): int(np.searchsorted(c, lv) + 1) for lv in levels}


def nearest(queries: np.ndarray, base: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(distance, index) of the nearest row of base for each query, exact."""
    b = torch.from_numpy(base.astype(np.float32))
    bsq = (b * b).sum(1)
    dist, idx = [], []
    for s in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[s:s + chunk].astype(np.float32))
        d2 = (q * q).sum(1, keepdim=True) + bsq[None] - 2 * q @ b.T
        v, i = d2.min(1)
        dist.append(v.clamp_min(0).sqrt().numpy())
        idx.append(i.numpy())
    return np.concatenate(dist), np.concatenate(idx)


def run() -> dict:
    cube = idest.curved_cube(10, 10_000, seed=10)
    x_tr, y_tr = common.load_fashion("train")
    x_te, y_te = common.load_fashion("test")
    s_cube, s_px = pca_spectrum(cube), pca_spectrum(x_te.astype(np.float64))
    pca = {"curved_cube_10": {"components": components_for(s_cube), "top50_share": s_cube[:50].round(5).tolist()},
           "fashion_pixels": {"components": components_for(s_px), "top50_share": s_px[:50].round(5).tolist()}}
    print("PCA components for 90/95/99%:", pca["curved_cube_10"]["components"], pca["fashion_pixels"]["components"])

    rng = np.random.default_rng(0)
    same, diff = [], []
    while len(same) < N_PAIRS or len(diff) < N_PAIRS:
        i, j = (int(v) for v in rng.choice(len(x_te), 2, replace=False))
        bucket = same if y_te[i] == y_te[j] else diff
        if len(bucket) < N_PAIRS:
            bucket.append((i, j))
    real_d, _ = nearest(x_te[:2000], x_tr)
    out = {"real_test_to_train": {"n": 2000, "median": round(float(np.median(real_d)), 3),
                                  "p90": round(float(np.percentile(real_d, 90)), 3)}}
    show = {}
    for kind, pairs in (("same_class", same), ("different_class", diff)):
        a, b = x_te[[i for i, _ in pairs]], x_te[[j for _, j in pairs]]
        mid = (a + b) / 2
        dm, im = nearest(mid, x_tr)
        da, _ = nearest(a, x_tr)
        out[kind] = {"pairs": len(pairs), "midpoint_median": round(float(np.median(dm)), 3),
                     "endpoint_median": round(float(np.median(da)), 3),
                     "share_midpoint_farther_than_real_p90": round(float(np.mean(dm > np.percentile(real_d, 90))), 3),
                     "midpoint_nearest_same_class_as_an_end": round(float(np.mean(
                         (y_tr[im] == y_te[[i for i, _ in pairs]]) | (y_tr[im] == y_te[[j for _, j in pairs]]))), 3)}
        show[kind] = [{"a": int(i), "b": int(j), "nearest_train": int(im[k]), "dist_mid": round(float(dm[k]), 3),
                       "label_a": common.FASHION_CLASSES[y_te[i]], "label_b": common.FASHION_CLASSES[y_te[j]],
                       "label_nearest": common.FASHION_CLASSES[y_tr[im[k]]]}
                      for k, (i, j) in enumerate(pairs[:SHOW])]
        print(f"{kind}: midpoint to nearest train {out[kind]['midpoint_median']} (median), "
              f"endpoints {out[kind]['endpoint_median']}")
    print("real test image to nearest train (median):", out["real_test_to_train"]["median"])
    return {"pca": pca, "midpoints": out, "show": show}


def fig_manifold(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_tr, _ = common.load_fashion("train")
    x_te, _ = common.load_fashion("test")
    rows = r["show"]["same_class"][:2] + r["show"]["different_class"][:2]
    fig = plt.figure(figsize=(10.5, 5))
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 0.35, 4.2])
    for k, row in enumerate(rows):
        a, b = x_te[row["a"]], x_te[row["b"]]
        imgs = [a, (a + b) / 2, x_tr[row["nearest_train"]], b]
        titles = [row["label_a"], "midpoint", "nearest real", row["label_b"]]
        for c in range(4):
            ax = fig.add_subplot(gs[k, c])
            ax.imshow(imgs[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[c], fontsize=8)
    ax = fig.add_subplot(gs[:, 5])
    for key, lab, col in (("fashion_pixels", "FashionMNIST pixels", "#1d6fb8"), ("curved_cube_10", "curved 10-d cube", "#b7410e")):
        ax.plot(np.arange(1, 51), np.cumsum(r["pca"][key]["top50_share"]), color=col, label=lab)
    ax.axhline(0.9, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("principal components")
    ax.set_ylabel("share of variance held")
    ax.set_title("PCA: variance held by the first components", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("manifold.py")
    torch.set_num_threads(4)
    res = run()
    res["guard"] = guard.report()
    common.write_results("manifold", res)
    if args.figures:
        fig_manifold(res, common.FIGURES / "manifold.png")


if __name__ == "__main__":
    main()
