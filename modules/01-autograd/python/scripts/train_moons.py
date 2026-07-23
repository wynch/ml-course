"""Train the from-scratch MLP on sklearn's make_moons to convergence.

Produces two figures:
  figures/loss_curve.png       - training loss vs step
  figures/decision_boundary.png - the learned classifier over the plane

Run:  uv run scripts/train_moons.py
"""

import _bootstrap  # noqa: F401  (sets sys.path + FIGDIR)
from _bootstrap import FIGDIR

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP


def make_loss(model, X, y):
    """Mean squared-margin (SVM-ish) loss + L2 regularization, over the batch."""
    inputs = [[Value(xi) for xi in row] for row in X]
    scores = [model(x) for x in inputs]
    # svm "max-margin" loss: relu(1 - y*score)
    losses = [(1 + -yi * s).relu() for yi, s in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg_loss
    # accuracy (plain python, no grad needed)
    acc = [(yi > 0) == (s.data > 0) for yi, s in zip(y, scores)]
    return total, sum(acc) / len(acc)


def main() -> None:
    np.random.seed(1337)
    import random

    random.seed(1337)

    X, y = make_moons(n_samples=100, noise=0.1, random_state=1337)
    y = y * 2 - 1  # map {0,1} -> {-1,+1}

    model = MLP(2, [16, 16, 1], nonlin="tanh")
    print(f"model: {model}")
    print(f"number of parameters: {len(model.parameters())}")

    steps = 100
    history = []
    for k in range(steps):
        total_loss, acc = make_loss(model, X, y)
        model.zero_grad()
        total_loss.backward()
        # simple SGD with decaying learning rate
        lr = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append((k, total_loss.data, acc))
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k:3d}  loss {total_loss.data:.4f}  acc {acc*100:.1f}%")

    final_loss, final_acc = history[-1][1], history[-1][2]
    print(f"\nFINAL  loss {final_loss:.4f}  acc {final_acc*100:.1f}%")

    # ---- figure 1: loss curve --------------------------------------------
    ks = [h[0] for h in history]
    ls = [h[1] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ls, color="#c1121f", lw=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"make_moons training loss (final acc {final_acc*100:.0f}%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "loss_curve.png", dpi=100)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'loss_curve.png'}")

    # ---- figure 2: decision boundary -------------------------------------
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = [model([Value(a), Value(b)]).data for a, b in grid]
    Z = np.array(scores).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#a8dadc", "#f1a7a1"], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=30)
    ax.set_title("Decision boundary of the from-scratch MLP")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(FIGDIR / "decision_boundary.png", dpi=100)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'decision_boundary.png'}")


if __name__ == "__main__":
    main()
