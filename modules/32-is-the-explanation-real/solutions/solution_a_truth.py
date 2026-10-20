"""Solution (a): best cosines and one-to-one seed matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=0, keepdims=True)
    return m / np.where(n == 0, 1, n)


def best_cosines(truth: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return (unit(truth).T @ unit(dec)).max(1)


def seed_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = unit(a).T @ unit(b)
    rows, cols = linear_sum_assignment(-c)
    out = np.empty(a.shape[1])
    out[rows] = c[rows, cols]
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
    from common import ARTIFACTS
    from model import load

    W = load(ARTIFACTS / "sparse.pt").blocks[1].mlp.W_out.weight.detach().numpy()
    rng = np.random.default_rng(0)
    shuffled = W[:, rng.permutation(W.shape[1])] * np.where(rng.random(W.shape[1]) < 0.5, -1, 1)
    # A flipped column has cosine -1 with its twin: best_cosines finds another column, seed_match pairs it at -1.
    print("MMCS of W_out against itself, shuffled and half flipped:", best_cosines(W, shuffled).mean())
    print("share matched one to one at cosine >= 0.7:", (seed_match(W, shuffled) >= 0.7).mean())
    noise = rng.normal(size=(W.shape[0], 2048))
    print("against 2,048 random directions:", best_cosines(W, noise).mean())
