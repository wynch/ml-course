"""Exercise (b) — add a second hidden layer and compare training curves.

The one-hidden-layer MLP (784 -> 256 -> 10) is built for you below. Your job:
build a two-hidden-layer network and train it on the same data, then look at
the comparison figure to see whether the extra depth helps here.

Run:
    uv run python exercises/ex_b_deeper_net.py

Writes figures/ex_b_depth_comparison.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_fashion_mnist
from src.nn import MLP, accuracy
from src.optim import SGD
from src.train import train, train_val_split

FIG = Path(__file__).resolve().parents[2] / "figures"


def run(sizes, x_tr, y_tr, x_val, y_val, seed=0):
    rng = np.random.default_rng(seed)
    model = MLP(sizes, rng)
    opt = SGD(model.params_and_grads, lr=0.1, momentum=0.9)
    hist = train(model, opt, x_tr, y_tr, x_val, y_val,
                 epochs=15, batch_size=128, rng=rng, verbose=False)
    return model, hist


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_fashion_mnist(train_subsample=15000, seed=0)
    rng = np.random.default_rng(0)
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, 0.1, rng)

    # one hidden layer (given)
    m1, h1 = run([784, 256, 10], x_tr, y_tr, x_val, y_val)

    # TODO(you): build a TWO-hidden-layer network and train it the same way.
    #   Hint: MLP takes a list of layer sizes; insert a second hidden width,
    #   e.g. [784, 256, 128, 10]. Then call run(...) with it.
    m2, h2 = None, None  # replace this line
    raise NotImplementedError("build and train the two-hidden-layer network")

    # ---- comparison figure (works once m2/h2 are filled in) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ep = np.arange(1, len(h1.val_loss) + 1)
    ax1.plot(ep, h1.val_loss, label="1 hidden", lw=2)
    ax1.plot(ep, h2.val_loss, label="2 hidden", lw=2)
    ax1.set_title("val loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, h1.val_acc, label="1 hidden", lw=2)
    ax2.plot(ep, h2.val_acc, label="2 hidden", lw=2)
    ax2.set_title("val accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Depth comparison on FashionMNIST")
    fig.savefig(FIG / "ex_b_depth_comparison.png")
    plt.close(fig)

    print(f"1-hidden test acc: {accuracy(m1, x_test, y_test):.4f}")
    print(f"2-hidden test acc: {accuracy(m2, x_test, y_test):.4f}")


if __name__ == "__main__":
    main()
