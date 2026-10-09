"""Gaussian naive Bayes from scratch: 2D first, then 784 pixels.

Part 1 fits the naive (diagonal) model and the full-covariance model to the same
two tilted 2D Gaussians and draws both decision regions, so you can see exactly
what the independence assumption throws away.

Part 2 runs the naive model on two-class FashionMNIST subsets with 784 features:
an easy pair (Trouser vs Sneaker) and a hard one (Pullover vs Coat). It falls
back to a synthetic stand-in if the dataset is not cached.

Produces: figures/naive_bayes_regions.png, figures/naive_bayes_fashion.png

Run:  uv run scripts/naive_bayes.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import numpy as np

from origins.bayes import GaussianBayes, GaussianNaiveBayes
from origins.data import (
    FASHION_CLASSES,
    fashion_available,
    gaussian_pair_2d,
    load_fashion_mnist,
    take_classes,
    two_gaussians,
)
from origins.plots import ACCENT, MUTED, T0, WARN, finish, plt

EASY_PAIR = (1, 7)   # Trouser vs Sneaker — different silhouettes
HARD_PAIR = (2, 4)   # Pullover vs Coat  — near-identical silhouettes
N_TRAIN_FASHION = 30_000
VAR_SMOOTHING = 1e-2


def grid(X, pad=1.0, n=320):
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    G0, G1 = np.meshgrid(x0, x1)
    return G0, G1, np.column_stack([G0.ravel(), G1.ravel()])


def part1() -> dict:
    Xtr, ytr = gaussian_pair_2d(400, seed=3)
    Xte, yte = gaussian_pair_2d(2000, seed=11)

    nb = GaussianNaiveBayes().fit(Xtr, ytr)
    fb = GaussianBayes().fit(Xtr, ytr)
    acc_nb, acc_fb = nb.score(Xte, yte), fb.score(Xte, yte)
    print("2D tilted Gaussians (400 train / 2000 test)")
    print(f"  naive Bayes (diagonal)      test accuracy {acc_nb:.4f}")
    print(f"  full covariance (quadratic) test accuracy {acc_fb:.4f}")
    print(f"  naive means  class0 {nb.theta_[0].round(3)}  class1 {nb.theta_[1].round(3)}")
    print(f"  naive vars   class0 {nb.var_[0].round(3)}  class1 {nb.var_[1].round(3)}")
    print(f"  true corr    class0 {np.corrcoef(Xtr[ytr == 0].T)[0, 1]:+.3f}  "
          f"class1 {np.corrcoef(Xtr[ytr == 1].T)[0, 1]:+.3f}")

    G0, G1, P = grid(Xtr)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, model, name, acc in (
        (axes[0], nb, "naive Bayes — diagonal variances", acc_nb),
        (axes[1], fb, "full covariance — tilted ellipses", acc_fb),
    ):
        Z = model.predict(P).reshape(G0.shape)
        ax.contourf(G0, G1, Z, levels=[-0.5, 0.5, 1.5], colors=["#f0e3c8", "#d9ecea"])
        ax.contour(G0, G1, Z, levels=[0.5], colors=[WARN], linewidths=1.6)
        for c, col, mark in ((0, T0, "o"), (1, ACCENT, "^")):
            m = ytr == c
            ax.scatter(Xtr[m, 0], Xtr[m, 1], s=11, c=col, marker=mark, lw=0, alpha=0.85)
        ax.set_title(f"{name}\ntest accuracy {acc:.3f}")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")

    ax = axes[2]
    prob1 = nb.predict_proba(P)[:, 1].reshape(G0.shape)
    im = ax.contourf(G0, G1, prob1, levels=np.linspace(0, 1, 21), cmap="viridis")
    ax.contour(G0, G1, prob1, levels=[0.5], colors="white", linewidths=1.4)
    fig.colorbar(im, ax=ax, label="P(class 1 | x)  — naive model")
    ax.set_title("the posterior, not just the argmax")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

    fig.tight_layout()
    finish(fig, FIGDIR / "naive_bayes_regions.png")
    return {"acc_nb_2d": acc_nb, "acc_fb_2d": acc_fb}


def fit_pair(xtr, ytr_all, xte, yte_all, pair, names):
    Xtr, ytr = take_classes(xtr, ytr_all, pair)
    Xte, yte = take_classes(xte, yte_all, pair)
    nb = GaussianNaiveBayes(var_smoothing=VAR_SMOOTHING).fit(Xtr, ytr)
    acc = nb.score(Xte, yte)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(yte, nb.predict(Xte)):
        cm[t, p] += 1
    print(f"  {names[0]} vs {names[1]}: train {len(Xtr)}  test {len(Xte)}  "
          f"accuracy {acc:.4f}  confusion {cm.tolist()}")
    return nb, acc, cm


def part2() -> dict:
    if fashion_available():
        source = "FashionMNIST"
        xtr, ytr_all, xte, yte_all = load_fashion_mnist(train_subsample=N_TRAIN_FASHION, seed=0)
        pairs = [
            (HARD_PAIR, [FASHION_CLASSES[c] for c in HARD_PAIR]),
            (EASY_PAIR, [FASHION_CLASSES[c] for c in EASY_PAIR]),
        ]
    else:
        source = "synthetic 784-d Gaussians (FashionMNIST not cached)"
        xtr, ytr_all = two_gaussians(N_TRAIN_FASHION, sep=0.9, seed=0, dim=784)
        xte, yte_all = two_gaussians(4000, sep=0.9, seed=1, dim=784)
        pairs = [((0, 1), ["class 0", "class 1"]), ((0, 1), ["class 0", "class 1"])]

    print(f"\n{source} — Gaussian naive Bayes on 784 raw pixels "
          f"(var_smoothing={VAR_SMOOTHING:g})")
    fits = [fit_pair(xtr, ytr_all, xte, yte_all, p, n) for p, n in pairs]
    print(f"  parameters per pair: 2 classes x (784 means + 784 variances) = "
          f"{4 * xtr.shape[1]}, fitted in closed form (no optimiser)")

    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.0))
    for r, ((_, names), (nb, acc, cm)) in enumerate(zip(pairs, fits)):
        for j in range(2):
            ax = axes[r, j]
            ax.imshow(nb.theta_[j].reshape(28, 28), cmap="viridis")
            ax.set_title(f"mean pixel — {names[j]}", fontsize=9)
            ax.axis("off")
        ax = axes[r, 2]
        diff = nb.theta_[1] - nb.theta_[0]
        lim = np.abs(diff).max()
        im = ax.imshow(diff.reshape(28, 28), cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title("mean difference", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[r, 3]
        ax.imshow(cm, cmap="viridis")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] < cm.max() * 0.6 else "black")
        ax.set_xticks([0, 1], [f"pred\n{n}" for n in names], fontsize=7)
        ax.set_yticks([0, 1], [f"true\n{n}" for n in names], fontsize=7)
        ax.set_title(f"accuracy {acc:.4f}", fontsize=9, color=T0 if acc < 0.9 else ACCENT)
        ax.grid(False)

    fig.suptitle(
        f"Gaussian naive Bayes on {source}: same 3,136 parameters, two very different pairs",
        fontsize=10,
    )
    fig.tight_layout()
    finish(fig, FIGDIR / "naive_bayes_fashion.png")
    return {
        "source": source,
        "acc_hard": fits[0][1],
        "acc_easy": fits[1][1],
        "cm_hard": fits[0][2].tolist(),
        "cm_easy": fits[1][2].tolist(),
    }


def main() -> None:
    r1 = part1()
    r2 = part2()
    print("\nsummary:", {**r1, **r2})


if __name__ == "__main__":
    main()
