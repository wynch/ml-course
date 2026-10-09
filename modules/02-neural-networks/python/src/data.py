"""Data loading: 2D toy datasets and FashionMNIST via Hugging Face `datasets`.

The toy datasets (make_moons, make_circles) are 2D so we can *see* the decision
boundary. FashionMNIST is real image data: 28x28 grayscale clothing photos in
10 classes, the drop-in "harder MNIST" that a linear model cannot ace.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons

# FashionMNIST class names, in label order 0..9.
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def toy_dataset(kind: str, n: int = 600, noise: float = 0.2, seed: int = 0):
    """Return (X, y) for a 2D toy problem: 'moons' or 'circles'."""
    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif kind == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed)
    else:
        raise ValueError(f"unknown toy dataset {kind!r}")
    # standardize features so the net trains on a sane scale
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64), y.astype(np.int64)


def load_fashion_mnist(train_subsample: int | None = None, seed: int = 0):
    """Load FashionMNIST through Hugging Face `datasets`.

    Returns (x_train, y_train, x_test, y_test) with images flattened to 784
    floats in [0, 1]. The full test set is always returned so accuracy is
    reported honestly; only the *training* set may be subsampled for speed.
    """
    from datasets import load_dataset

    # canonical FashionMNIST now lives under the zalando-datasets namespace
    ds = load_dataset("zalando-datasets/fashion_mnist")

    def to_arrays(split):
        # images come as PIL objects; stack into (N, 28, 28) then flatten
        imgs = np.stack([np.asarray(im, dtype=np.float32) for im in split["image"]])
        x = imgs.reshape(len(imgs), -1) / 255.0
        y = np.asarray(split["label"], dtype=np.int64)
        return x, y

    x_train, y_train = to_arrays(ds["train"])
    x_test, y_test = to_arrays(ds["test"])

    if train_subsample is not None and train_subsample < len(x_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train), size=train_subsample, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]

    return x_train, y_train, x_test, y_test
