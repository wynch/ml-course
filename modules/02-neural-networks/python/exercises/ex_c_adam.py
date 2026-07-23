"""Exercise (c) — implement Adam and compare it against SGD.

A skeleton ``MyAdam`` optimizer is below with the state set up for you. Fill in
``step`` following the Adam update rule, then run the script to produce a figure
comparing the training curves of SGD vs your Adam on FashionMNIST.

Run:
    uv run python exercises/ex_c_adam.py

Writes figures/ex_c_adam_vs_sgd.png.

------------------------------------------------------------------------------
Adam update (per parameter, at step t):
    m <- b1*m + (1-b1)*g            # 1st moment (mean of gradients)
    v <- b2*v + (1-b2)*g*g          # 2nd moment (uncentered variance)
    m_hat = m / (1 - b1**t)         # bias correction
    v_hat = v / (1 - b2**t)
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)
------------------------------------------------------------------------------
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


class MyAdam:
    def __init__(self, get_fn, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._get = get_fn
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m, self._v, self._t = {}, {}, 0

    def step(self):
        self._t += 1
        for i, (p, g) in enumerate(self._get()):
            m = self._m.get(i, np.zeros_like(p))
            v = self._v.get(i, np.zeros_like(p))
            # TODO(you): implement the Adam update.
            #   1. update the moving averages m and v from the gradient g
            #   2. store them back into self._m[i] / self._v[i]
            #   3. bias-correct with (1 - beta**t)
            #   4. update p in place: p -= lr * m_hat / (sqrt(v_hat) + eps)
            raise NotImplementedError("implement the Adam step")


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
    m_adam, h_adam = run(lambda g: MyAdam(g, lr=1e-3),
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

    print(f"SGD  test acc: {accuracy(m_sgd, x_test, y_test):.4f}")
    print(f"Adam test acc: {accuracy(m_adam, x_test, y_test):.4f}")


if __name__ == "__main__":
    main()
