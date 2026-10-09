"""PCA by power iteration, cross-checked against numpy.linalg.eigh.

Part 1 is 2-D and visual: one correlated cloud, the power-iteration vector
swinging onto the long axis over a handful of steps, and the 1-sigma variance
ellipse the two eigenvectors define.

Part 2 is 784-D: PCA on a FashionMNIST subset (synthetic fallback if the dataset
is not cached). Scree plot, the components rendered as images, the 2-D
projection coloured by class, and reconstruction error against the number of
components kept. Every eigenpair we compute by power iteration is compared to
``numpy.linalg.eigh`` and the disagreement is printed, not assumed.

Produces: figures/pca_power_iteration.png, figures/pca_scree.png,
          figures/pca_eigenimages.png, figures/pca_projection.png,
          figures/pca_reconstruction.png, figures/pca_recon_error.png
          python/pca_results.json  (inlined by the explorable)

Run:  uv run scripts/pca_lab.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import json

import numpy as np

from origins.data import (
    FASHION_CLASSES,
    anisotropic_blob,
    fashion_available,
    load_fashion_mnist,
    two_gaussians,
)
from origins.pca import PCA, _fix_signs, power_iteration, top_eigenpairs, reconstruction_error
from origins.plots import ACCENT, MUTED, T0, WARN, finish, plt

N_TRAIN = 10_000
N_COMPONENTS = 40
K_RECON = [1, 2, 5, 10, 20, 50, 100, 200, 400, 784]
SHOW_ITERS = [0, 1, 2, 3]


# ─────────────────────────── part 1 · 2-D ───────────────────────────


def part1() -> dict:
    X = anisotropic_blob(400, seed=7)
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (len(X) - 1)

    lam1, v1, n1, hist = power_iteration(C, tol=1e-12, seed=0, trace=True)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Vp = _fix_signs(np.array([v1]))[0]
    Vr = _fix_signs(V.T)[0]
    angle = np.degrees(np.arccos(np.clip(abs(float(Vp @ Vr)), -1, 1)))

    print("Part 1 — 2D cloud (400 points, true stretch 3.0/0.8 rotated 30 degrees)")
    print(f"  covariance = {C.round(4).tolist()}")
    print(f"  power iteration converged in {n1} iterations (tol 1e-12)")
    print(f"  lambda1: power {lam1:.10f}   eigh {w[0]:.10f}   "
          f"rel diff {abs(lam1 - w[0]) / w[0]:.3e}")
    print(f"  eigenvector angle to eigh's: {angle:.3e} degrees")
    print(f"  eigenvalue gap lambda2/lambda1 = {w[1] / w[0]:.4f} "
          f"(the per-iteration shrink factor)")
    ang_true = np.degrees(np.arctan2(Vr[1], Vr[0])) % 180
    print(f"  PC1 direction {ang_true:.2f} degrees vs the 30 degrees it was built with")

    # angle of each traced iterate to the converged direction
    conv = np.array([
        np.degrees(np.arccos(np.clip(abs(float(h @ Vr / np.linalg.norm(h))), -1, 1)))
        for h in hist
    ])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=9, c=MUTED, lw=0, alpha=0.45)
    cmap = plt.get_cmap("viridis")
    for j, it in enumerate(SHOW_ITERS):
        h = hist[min(it, len(hist) - 1)]
        h = h if h @ Vr >= 0 else -h
        s = 4.6
        col = cmap(0.08 + 0.84 * j / (len(SHOW_ITERS) - 1))
        ax.annotate("", xy=(h[0] * s, h[1] * s), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=col))
        ax.text(h[0] * s * 1.10, h[1] * s * 1.10, f"it {it}", fontsize=7.5,
                color=col, ha="center", va="center")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-6.0, 6.0)
    ax.set_aspect("equal")
    ax.set_title(f"(a) power iteration swings onto PC1\n"
                 f"random start is {conv[0]:.0f}° off; 3 steps get within {conv[3]:.3f}°")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=9, c=MUTED, lw=0, alpha=0.45)
    t = np.linspace(0, 2 * np.pi, 200)
    ell = (V * np.sqrt(w)) @ np.array([np.cos(t), np.sin(t)])
    ax.plot(ell[0], ell[1], color=WARN, lw=1.6, label="1σ variance ellipse")
    for i, (col, lab) in enumerate(((T0, "PC1"), (ACCENT, "PC2"))):
        d = Vr if i == 0 else _fix_signs(V.T)[1]
        s = 2.2 * np.sqrt(w[i])
        ax.annotate("", xy=(d[0] * s, d[1] * s), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2.4, color=col))
        ax.text(d[0] * s * 1.12, d[1] * s * 1.12,
                f"{lab}  λ={w[i]:.2f}", fontsize=8, color=col)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) the eigenvectors, scaled by √λ")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    ax = axes[2]
    ax.semilogy(np.arange(len(conv)), np.maximum(conv, 1e-16), "o-", color=T0, lw=1.8, ms=4)
    ratio = w[1] / w[0]
    ref = conv[1] * ratio ** np.arange(len(conv) - 1)
    ax.semilogy(np.arange(1, len(conv)), np.maximum(ref, 1e-16), "--", color=MUTED, lw=1.2,
                label=f"geometric at (λ2/λ1)^k = {ratio:.3f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("angle to PC1  (degrees, log)")
    ax.set_title(f"(c) geometric convergence — {n1} iterations to 1e-12")
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish(fig, FIGDIR / "pca_power_iteration.png")

    return {
        "cov_2d": C.tolist(),
        "iters_2d": n1,
        "lam_power_2d": [lam1, float(w[1])],
        "lam_eigh_2d": w.tolist(),
        "angle_deg_2d": float(angle),
        "ratio_2d": float(ratio),
        "pc1_angle_deg": float(ang_true),
        "trace_2d": [h.tolist() for h in hist[:14]],
        "points_2d": X[:180].round(4).tolist(),
    }


# ─────────────────────────── part 2 · 784-D ───────────────────────────


def part2() -> dict:
    if fashion_available():
        source = "FashionMNIST"
        xtr, ytr, xte, yte = load_fashion_mnist(train_subsample=N_TRAIN, seed=0)
        names = FASHION_CLASSES
    else:
        source = "synthetic 784-d Gaussians (FashionMNIST not cached)"
        xtr, ytr = two_gaussians(N_TRAIN, sep=2.0, seed=0, dim=784)
        xte, yte = two_gaussians(4000, sep=2.0, seed=1, dim=784)
        names = ["class 0", "class 1"]
    print(f"\nPart 2 — PCA on {source}: {xtr.shape[0]} x {xtr.shape[1]}")

    # full spectrum with eigh (the reference)
    ref = PCA(784, solver="eigh").fit(xtr)
    evr = ref.explained_variance_ratio_
    cum = np.cumsum(evr)
    for target in (0.5, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(cum, target) + 1)
        print(f"  {int(target * 100)}% of the variance needs {k:3d} of 784 components")
    print(f"  PC1 alone explains {100 * evr[0]:.2f}% of the total variance "
          f"({ref.total_variance_:.4f} summed over all 784 pixels)")

    # the same top-k by power iteration, then compare
    ours = PCA(N_COMPONENTS, solver="power", tol=1e-10, seed=0).fit(xtr)
    lam_ref = ref.explained_variance_[:N_COMPONENTS]
    lam_ours = ours.explained_variance_
    rel = np.abs(lam_ours - lam_ref) / lam_ref
    cosang = np.abs((ours.components_ * ref.components_[:N_COMPONENTS]).sum(axis=1))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    print(f"  power iteration + deflation, {N_COMPONENTS} components:")
    print(f"    iterations per component: min {min(ours.n_iter_)}  "
          f"median {int(np.median(ours.n_iter_))}  max {max(ours.n_iter_)}")
    print(f"    eigenvalue relative error vs eigh: max {rel.max():.3e}  "
          f"mean {rel.mean():.3e}")
    print(f"    eigenvector angle vs eigh (deg):   max {ang.max():.4f}  "
          f"mean {ang.mean():.4f}")
    print(f"    first 5 angles (deg): {ang[:5].round(6).tolist()}")
    print(f"    last  5 angles (deg): {ang[-5:].round(4).tolist()}")

    # ── scree ──
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    ax.bar(np.arange(1, N_COMPONENTS + 1), 100 * evr[:N_COMPONENTS], color=T0, width=0.8)
    ax.set_xlabel("component")
    ax.set_ylabel("% of total variance")
    ax.set_title(f"scree — first {N_COMPONENTS} of 784 components")
    ax = axes[1]
    ax.plot(np.arange(1, 785), 100 * cum, color=ACCENT, lw=1.8)
    for target, col in ((0.9, WARN), (0.95, MUTED)):
        k = int(np.searchsorted(cum, target) + 1)
        ax.axhline(100 * target, color=col, ls=":", lw=1.0)
        ax.plot([k], [100 * cum[k - 1]], "o", color=col, ms=5)
        ax.annotate(f"{int(target * 100)}% at k = {k}", xy=(k, 100 * cum[k - 1]),
                    xytext=(k + 60, 100 * target - 9), fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.set_xlabel("components kept, k")
    ax.set_ylabel("cumulative % of variance")
    ax.set_ylim(0, 103)
    ax.set_title("cumulative variance — the compression curve")
    fig.suptitle(f"PCA spectrum of {source}", fontsize=10)
    fig.tight_layout()
    finish(fig, FIGDIR / "pca_scree.png")

    # ── components as images ──
    fig, axes = plt.subplots(2, 8, figsize=(11.2, 3.6))
    mean_img = ref.mean_.reshape(28, 28)
    axes[0, 0].imshow(mean_img, cmap="gray")
    axes[0, 0].set_title("mean", fontsize=8)
    axes[0, 0].axis("off")
    for i in range(15):
        ax = axes.ravel()[i + 1]
        comp = ref.components_[i].reshape(28, 28)
        lim = np.abs(comp).max()
        ax.imshow(comp, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(f"PC{i + 1} · {100 * evr[i]:.1f}%", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "The first 15 eigenvectors of the pixel covariance, drawn as 28x28 images "
        "(red = add, blue = subtract)", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    finish(fig, FIGDIR / "pca_eigenimages.png")

    # ── 2D projection ──
    Z = ref.transform(xtr)[:, :2]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    n_cls = len(names)
    cmap = plt.get_cmap("viridis", n_cls)
    for c in range(n_cls):
        m = ytr == c
        ax.scatter(Z[m, 0], Z[m, 1], s=5, color=cmap(c), lw=0, alpha=0.55, label=names[c])
    ax.set_xlabel(f"PC1  ({100 * evr[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2  ({100 * evr[1]:.1f}% of variance)")
    ax.set_title(f"{source} projected onto its top two principal components\n"
                 "(no labels were used to find these axes)")
    ax.legend(fontsize=7, markerscale=2.2, ncol=2, loc="upper right")
    fig.tight_layout()
    finish(fig, FIGDIR / "pca_projection.png")

    # ── reconstruction grid ──
    rng = np.random.default_rng(4)
    picks = rng.choice(len(xte), size=6, replace=False)
    ks = [1, 2, 5, 10, 20, 50, 100, 784]
    fig, axes = plt.subplots(len(ks) + 1, 6, figsize=(7.4, 1.15 * (len(ks) + 1)))
    for j, p in enumerate(picks):
        axes[0, j].imshow(xte[p].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("original")
    axes[0, 0].text(-0.35, 0.5, "original", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=8)
    for i, k in enumerate(ks):
        Xr = ref.reconstruct(xte[picks], k=k)
        mse = reconstruction_error(xte, ref, k)
        for j in range(6):
            axes[i + 1, j].imshow(Xr[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[i + 1, j].axis("off")
        axes[i + 1, 0].text(-0.35, 0.5, f"k={k}\nmse {mse:.4f}",
                            transform=axes[i + 1, 0].transAxes,
                            rotation=90, va="center", ha="center", fontsize=7)
    fig.suptitle(f"{source} reconstructed from k components", fontsize=10)
    fig.tight_layout()
    finish(fig, FIGDIR / "pca_reconstruction.png")

    # ── reconstruction error vs k ──
    errs_tr = [reconstruction_error(xtr, ref, k) for k in K_RECON]
    errs_te = [reconstruction_error(xte, ref, k) for k in K_RECON]
    tail = [float(ref.explained_variance_[k:].sum() * (len(xtr) - 1) / len(xtr) / 784)
            for k in K_RECON]
    print("\n  reconstruction MSE per pixel (held-out test set)")
    for k, a, b in zip(K_RECON, errs_tr, errs_te):
        print(f"    k = {k:4d}   train {a:.6f}   test {b:.6f}   "
              f"({100 * (1 - b / errs_te[0]):5.1f}% below k=1)")

    # k = 784 reconstructs exactly (mse ~1e-31), which no log axis can show —
    # plot the truncations and state the exact case in words.
    kp = [k for k in K_RECON if k < 784]
    m = len(kp)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.loglog(kp, errs_tr[:m], "o-", color=MUTED, lw=1.4, ms=4, label="train MSE")
    ax.loglog(kp, errs_te[:m], "o-", color=T0, lw=2.0, ms=5, label="held-out test MSE")
    ax.loglog(kp, tail[:m], "--", color=ACCENT, lw=1.4,
              label="discarded eigenvalue tail / 784")
    ax.text(1.15, errs_te[-2] * 1.5,
            f"k = 784 reconstructs exactly:\nMSE {errs_te[-1]:.1e}",
            fontsize=8, color=MUTED)
    ax.set_xlabel("components kept, k  (log)")
    ax.set_ylabel("mean squared error per pixel  (log)")
    ax.set_title("Reconstruction error is exactly the variance you threw away")
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish(fig, FIGDIR / "pca_recon_error.png")

    return {
        "source": source,
        "n_train": int(len(xtr)),
        "evr": evr[:120].round(8).tolist(),
        "cum": cum[:784].round(8).tolist(),
        "k90": int(np.searchsorted(cum, 0.9) + 1),
        "k95": int(np.searchsorted(cum, 0.95) + 1),
        "k99": int(np.searchsorted(cum, 0.99) + 1),
        "total_variance": float(ref.total_variance_),
        "power_iters": ours.n_iter_,
        "rel_eig_err_max": float(rel.max()),
        "rel_eig_err_mean": float(rel.mean()),
        "angle_max_deg": float(ang.max()),
        "angle_mean_deg": float(ang.mean()),
        "k_recon": K_RECON,
        "mse_test": [float(v) for v in errs_te],
        "mse_train": [float(v) for v in errs_tr],
        "class_names": names,
    }


def main() -> None:
    r1 = part1()
    r2 = part2()
    out = FIGDIR.parent / "python" / "pca_results.json"
    out.write_text(json.dumps({**r1, **r2}, indent=1))
    print(f"\n  wrote {out.name} (scree + trace data inlined by the explorable)")


if __name__ == "__main__":
    main()
