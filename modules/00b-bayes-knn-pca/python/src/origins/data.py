"""Datasets for module 00b.

Two families:

* **Seeded 2D Gaussians** — the point of these is that we *know* the generating
  distribution, so we can compute the Bayes error in closed form and check an
  empirical classifier against it. Nothing else in the course gives you that.
* **FashionMNIST** — 28x28 grayscale clothing photos, loaded through Hugging
  Face ``datasets`` from the local cache. Used for the two-class naive Bayes and
  for PCA on 784 dimensions.
"""

from __future__ import annotations

import numpy as np

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


# ───────────────────────────── synthetic ─────────────────────────────


def two_gaussians(
    n: int,
    *,
    sep: float = 2.0,
    sigma: float = 1.0,
    seed: int = 0,
    dim: int = 2,
):
    """Two spherical Gaussians with equal priors, separated along axis 0.

    Class 0 sits at ``(-sep/2, 0, ...)``, class 1 at ``(+sep/2, 0, ...)``, both
    with covariance ``sigma**2 * I``. Equal priors, so a fair coin picks the
    class of each point.

    The whole reason for this shape: with equal spherical covariances the
    optimal (Bayes) decision rule is the vertical line ``x0 = 0``, and its error
    rate is ``Phi(-sep / (2*sigma))`` exactly — see
    :func:`origins.knn.bayes_error_two_gaussians`.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    mu = np.zeros((2, dim))
    mu[0, 0] = -sep / 2.0
    mu[1, 0] = +sep / 2.0
    X = rng.normal(loc=mu[y], scale=sigma, size=(n, dim))
    return X.astype(np.float64), y.astype(np.int64)


def anisotropic_blob(n: int = 400, *, seed: int = 7):
    """A single correlated 2D cloud — the standard "what does PCA see" picture.

    Built by stretching an isotropic Gaussian by (3.0, 0.8) and rotating it 30°,
    so the true principal directions are known before PCA is ever run.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    scale = np.array([3.0, 0.8])
    theta = np.deg2rad(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (z * scale) @ R.T


def _rot(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


#: (mean, scale, rotation-degrees) of the two classes in :func:`gaussian_pair_2d`.
PAIR_2D = (
    (np.array([-1.1, -0.4]), np.array([1.7, 0.5]), 40.0),
    (np.array([1.3, 0.5]), np.array([0.7, 1.6]), -25.0),
)


def gaussian_pair_2d(n: int = 400, *, seed: int = 3):
    """Two 2D Gaussians with different, **rotated** covariances.

    Both classes are correlated in x0/x1, which is precisely what naive Bayes
    assumes away. That makes this the honest demo set: the naive model has to
    draw axis-aligned ellipses through tilted clouds, and the figure shows what
    that costs against the full-covariance Bayes rule.
    """
    rng = np.random.default_rng(seed)
    counts = (n // 2, n - n // 2)
    parts, labels = [], []
    for c, (mean, scale, deg) in enumerate(PAIR_2D):
        z = rng.normal(size=(counts[c], 2)) * scale
        parts.append(z @ _rot(deg).T + mean)
        labels.append(np.full(counts[c], c, dtype=np.int64))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ───────────────────────────── FashionMNIST ─────────────────────────────


def load_fashion_mnist(*, train_subsample: int | None = None, seed: int = 0):
    """Load FashionMNIST via Hugging Face ``datasets``.

    Returns ``(x_train, y_train, x_test, y_test)`` with images flattened to 784
    floats in ``[0, 1]``. Reads the local cache (``~/.cache/huggingface``); the
    module scripts fall back to synthetic data when it is not there.
    """
    from datasets import load_dataset

    ds = load_dataset("zalando-datasets/fashion_mnist")

    def to_arrays(split):
        imgs = np.stack([np.asarray(im, dtype=np.float32) for im in split["image"]])
        return imgs.reshape(len(imgs), -1) / 255.0, np.asarray(split["label"], dtype=np.int64)

    x_train, y_train = to_arrays(ds["train"])
    x_test, y_test = to_arrays(ds["test"])

    if train_subsample is not None and train_subsample < len(x_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train), size=train_subsample, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]

    return x_train, y_train, x_test, y_test


def fashion_available() -> bool:
    """True if FashionMNIST is already in the local Hugging Face cache."""
    import pathlib

    root = pathlib.Path.home() / ".cache" / "huggingface" / "datasets"
    for name in ("zalando-datasets___fashion_mnist", "fashion_mnist"):
        hit = root / name
        if hit.is_dir() and any(hit.rglob("*.arrow")):
            return True
    return False


def take_classes(x, y, classes):
    """Subset ``(x, y)`` to the given labels, relabelled 0..len(classes)-1."""
    mask = np.isin(y, classes)
    xs, ys = x[mask], y[mask]
    remap = {c: i for i, c in enumerate(classes)}
    return xs, np.array([remap[int(v)] for v in ys], dtype=np.int64)
