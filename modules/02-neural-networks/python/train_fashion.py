"""Train an MLP on FashionMNIST to >=85% test accuracy, all from scratch.

Loads the data through Hugging Face `datasets`, flattens 28x28 -> 784, trains a
2-layer MLP with mini-batch SGD, and emits every figure the module promises:
sample grid, loss/accuracy curves, learned first-layer weight tiles, and a
confusion-matrix heatmap. Run:

    uv run python train_fashion.py

Test accuracy is always reported on the FULL 10k test set, even though we
subsample the training set for speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import FASHION_CLASSES, load_fashion_mnist
from src.nn import MLP, SoftmaxCrossEntropy, accuracy
from src.optim import SGD
from src.plots import (
    plot_confusion,
    plot_curves,
    plot_sample_grid,
    plot_weight_tiles,
)
from src.train import train, train_val_split

FIG = Path(__file__).resolve().parents[1] / "figures"


def main(seed: int = 0):
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("loading FashionMNIST via Hugging Face datasets ...")
    # subsample training to 20k for a fast demo; test stays full (10k)
    x_train, y_train, x_test, y_test = load_fashion_mnist(train_subsample=20000, seed=seed)
    print(f"train={x_train.shape} test={x_test.shape}")

    # sample grid (from the raw training data)
    plot_sample_grid(x_train, y_train, FASHION_CLASSES, FIG / "fashion_samples.png")

    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, val_frac=0.1, rng=rng)

    # 784 -> 256 -> 10
    model = MLP([784, 256, 10], rng)
    opt = SGD(model.params_and_grads, lr=0.1, momentum=0.9)

    hist = train(
        model, opt, x_tr, y_tr, x_val, y_val,
        epochs=25, batch_size=128, rng=rng,
        loss_fn=SoftmaxCrossEntropy(), verbose=True,
    )

    test_acc = accuracy(model, x_test, y_test)
    print(f"\nFINAL TEST ACCURACY (full 10k set): {test_acc:.4f}")

    # figures
    plot_curves(hist, "FashionMNIST — 784->256->10 MLP, SGD", FIG / "fashion_curves.png")
    W0 = model.layers[0].W  # first Linear layer weights (784, 256)
    plot_weight_tiles(W0, FIG / "fashion_weights.png")
    y_pred = model.predict(x_test)
    plot_confusion(y_test, y_pred, FASHION_CLASSES, FIG / "fashion_confusion.png")

    print("wrote: fashion_samples.png, fashion_curves.png, "
          "fashion_weights.png, fashion_confusion.png")
    return test_acc


if __name__ == "__main__":
    main()
