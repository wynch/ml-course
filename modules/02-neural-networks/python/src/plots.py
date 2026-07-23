"""Plotting helpers — every concept in this module produces a figure.

Kept separate from the training code so the training loop stays readable and
the visual style stays consistent across all the deliverables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})


def plot_curves(hist, title, path):
    """Loss and accuracy curves, train vs val, side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    epochs = np.arange(1, len(hist.train_loss) + 1)

    ax1.plot(epochs, hist.train_loss, label="train", lw=2)
    ax1.plot(epochs, hist.val_loss, label="val", lw=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, hist.train_acc, label="train", lw=2)
    ax2.plot(epochs, hist.val_acc, label="val", lw=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def plot_sample_grid(x, y, class_names, path, rows=4, cols=8, seed=0):
    """A grid of sample images with their class labels."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i].reshape(28, 28), cmap="gray")
        ax.set_title(class_names[y[i]], fontsize=7)
        ax.axis("off")
    fig.suptitle("FashionMNIST samples", y=1.01)
    fig.savefig(path)
    plt.close(fig)


def plot_weight_tiles(W, path, rows=8, cols=16):
    """Visualize the first-layer weight columns as 28x28 tiles.

    Each hidden unit reads all 784 pixels, so its incoming weights reshape to a
    28x28 image — a little template of what excites that unit.
    """
    n = min(rows * cols, W.shape[1])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.7))
    vmax = np.abs(W).max()
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(W[:, k].reshape(28, 28), cmap="seismic", vmin=-vmax, vmax=vmax)
    fig.suptitle("Learned first-layer weights (28x28 tiles)", y=1.01)
    fig.savefig(path, dpi=58)  # many small tiles -> keep dpi low to bound file size
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, path):
    """Confusion-matrix heatmap with row-normalized annotations."""
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(n_cls):
        for j in range(n_cls):
            v = cm_norm[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.5 else "black", fontsize=6,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path)
    plt.close(fig)


def decision_boundary_frame(model, X, y, ax, epoch, mesh):
    """Draw one decision-boundary frame onto ``ax`` (used by the GIF)."""
    xx, yy = mesh
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid)
    # probability of class 1 for a smooth filled contour
    from .nn import softmax

    z = softmax(probs)[:, 1].reshape(xx.shape)
    ax.clear()
    ax.contourf(xx, yy, z, levels=20, cmap="RdBu", alpha=0.7)
    ax.contour(xx, yy, z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=18, lw=0.4)
    ax.set_title(f"epoch {epoch + 1}")
    ax.set_xticks([])
    ax.set_yticks([])


def make_mesh(X, pad=0.5, step=0.02):
    """A dense coordinate grid covering the data, for boundary plots."""
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
    )
    return xx, yy
