"""Solution (c) — Adam vs SGD.

Uses the fully-implemented Adam from src/optim.py (the exercise asks you to
write your own MyAdam; this is the reference update).

Run:
    uv run python solutions/ex_c_adam.py
Writes figures/ex_c_adam_vs_sgd.png.
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
from src.optim import SGD, Adam
from src.train import train, train_val_split

FIG = Path(__file__).resolve().parents[2] / "figures"


def run(opt_factory, x_tr, y_tr, x_val, y_val, seed=0):
    rng = np.random.default_rng(seed)
    model = MLP([784, 256, 10], rng)
    opt = opt_factory(model.params_and_grads)
    hist = train(model, opt, x_tr, y_tr, x_val, y_val,
                 epochs=15, batch_size=128, rng=rng, verbose=False)
    return model, hist


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_fashion_mnist(train_subsample=15000, seed=0)
    rng = np.random.default_rng(0)
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, 0.1, rng)

    m_sgd, h_sgd = run(lambda g: SGD(g, lr=0.1, momentum=0.9),
                       x_tr, y_tr, x_val, y_val)
    m_adam, h_adam = run(lambda g: Adam(g, lr=1e-3),
                         x_tr, y_tr, x_val, y_val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ep = np.arange(1, len(h_sgd.train_loss) + 1)
    ax1.plot(ep, h_sgd.train_loss, label="SGD+momentum", lw=2)
    ax1.plot(ep, h_adam.train_loss, label="Adam", lw=2)
    ax1.set_title("train loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, h_sgd.val_acc, label="SGD+momentum", lw=2)
    ax2.plot(ep, h_adam.val_acc, label="Adam", lw=2)
    ax2.set_title("val accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Adam vs SGD on FashionMNIST")
    fig.savefig(FIG / "ex_c_adam_vs_sgd.png")
    plt.close(fig)

    a_sgd = accuracy(m_sgd, x_test, y_test)
    a_adam = accuracy(m_adam, x_test, y_test)
    print(f"SGD  test acc: {a_sgd:.4f}")
    print(f"Adam test acc: {a_adam:.4f}")
    print("Takeaway: Adam drives the training loss down fastest per epoch with no")
    print("lr tuning; well-tuned SGD+momentum reaches comparable final accuracy.")


if __name__ == "__main__":
    main()
