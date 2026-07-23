"""Train an MLP on 2D toy data and animate the decision boundary over epochs.

This produces the module's signature visual: a GIF showing the boundary bend
itself around the moons/circles as training proceeds. Run:

    uv run python train_toy.py

Writes figures/decision_boundary_moons.gif and figures/toy_circles.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.data import toy_dataset
from src.nn import MLP, SoftmaxCrossEntropy, accuracy
from src.optim import SGD
from src.plots import decision_boundary_frame, make_mesh
from src.train import train, train_val_split

FIG = Path(__file__).resolve().parents[1] / "figures"


def train_and_animate(kind: str, gif_path: Path, epochs: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = toy_dataset(kind, n=600, noise=0.2, seed=seed)
    x_tr, y_tr, x_val, y_val = train_val_split(X, y, val_frac=0.25, rng=rng)

    # 2 -> 16 -> 16 -> 2 : enough capacity for a curvy boundary
    model = MLP([2, 16, 16, 2], rng)
    opt = SGD(model.params_and_grads, lr=0.3, momentum=0.9)

    # a coarser mesh keeps the GIF small without changing the visible boundary
    mesh = make_mesh(X, step=0.04)
    snapshots = []  # store a light-weight copy of predictions per recorded epoch

    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=90)
    record_every = 4

    def on_epoch_end(epoch, m):
        if epoch % record_every == 0 or epoch == epochs - 1:
            # capture the current class-1 probability field for later playback
            from src.nn import softmax

            xx, yy = mesh
            grid = np.c_[xx.ravel(), yy.ravel()]
            z = softmax(m.forward(grid))[:, 1].reshape(xx.shape)
            snapshots.append((epoch, z))

    hist = train(
        model, opt, x_tr, y_tr, x_val, y_val,
        epochs=epochs, batch_size=32, rng=rng,
        loss_fn=SoftmaxCrossEntropy(), on_epoch_end=on_epoch_end, verbose=False,
    )

    xx, yy = mesh

    def draw(i):
        epoch, z = snapshots[i]
        ax.clear()
        ax.contourf(xx, yy, z, levels=12, cmap="RdBu", alpha=0.7)
        ax.contour(xx, yy, z, levels=[0.5], colors="k", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=18, lw=0.4)
        ax.set_title(f"{kind} — epoch {epoch + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(snapshots), interval=80)
    anim.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    final_acc = accuracy(model, X, y)
    print(f"[{kind}] frames={len(snapshots)} final full-set acc={final_acc:.3f} "
          f"val acc={hist.val_acc[-1]:.3f} -> {gif_path.name}")
    return final_acc


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train_and_animate("moons", FIG / "decision_boundary_moons.gif", epochs=120)

    # a static companion figure for circles (final boundary only)
    rng = np.random.default_rng(1)
    X, y = toy_dataset("circles", n=600, noise=0.15, seed=1)
    x_tr, y_tr, x_val, y_val = train_val_split(X, y, val_frac=0.25, rng=rng)
    model = MLP([2, 16, 16, 2], rng)
    opt = SGD(model.params_and_grads, lr=0.1, momentum=0.9)
    train(model, opt, x_tr, y_tr, x_val, y_val,
          epochs=120, batch_size=32, rng=rng, verbose=False)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    decision_boundary_frame(model, X, y, ax, 119, make_mesh(X))
    ax.set_title("circles — final boundary")
    fig.savefig(FIG / "toy_circles.png")
    plt.close(fig)
    print(f"[circles] final full-set acc={accuracy(model, X, y):.3f} -> toy_circles.png")


if __name__ == "__main__":
    main()
