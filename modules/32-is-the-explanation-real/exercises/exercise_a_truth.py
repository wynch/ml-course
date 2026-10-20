"""Exercise (a): score a dictionary against the truth, and two seeds against each other.

The sparse model's block-1 MLP has 2,048 neurons; column i of W_out is neuron
i's value vector. A dictionary has its own 2,048 decoder columns. Write:

1. best_cosines(truth, dec): for each true column, the highest cosine with any
   decoder column. Its mean is the MMCS of section 2.
2. seed_match(a, b): pair the columns of a and b one to one so that the summed
   cosine is largest (scipy.optimize.linear_sum_assignment does the search),
   and return each column of a's cosine with its partner.

Predict first: if you shuffle the columns of W_out and flip the sign of half
of them, what MMCS does W_out score against itself? What does seed_match give?

    uv run python ../exercises/exercise_a_truth.py      (from python/)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # noqa: F401


def unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=0, keepdims=True)
    return m / np.where(n == 0, 1, n)


def best_cosines(truth: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """truth (d, n_true), dec (d, n_dict) -> (n_true,)."""
    # TODO(you): normalise both, take all cosines, keep the best per true column.
    raise NotImplementedError


def seed_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b (d, n) -> (n,): cosine of each column of a with its one-to-one partner in b."""
    # TODO(you): cosine matrix, linear_sum_assignment on its negative, read the pairs back in a's order.
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
    from common import ARTIFACTS
    from model import load

    W = load(ARTIFACTS / "sparse.pt").blocks[1].mlp.W_out.weight.detach().numpy()
    rng = np.random.default_rng(0)
    shuffled = W[:, rng.permutation(W.shape[1])] * np.where(rng.random(W.shape[1]) < 0.5, -1, 1)
    print("MMCS of W_out against itself, shuffled and half flipped:", best_cosines(W, shuffled).mean())
    print("share matched one to one at cosine >= 0.7:", (seed_match(W, shuffled) >= 0.7).mean())
    noise = rng.normal(size=(W.shape[0], 2048))
    print("against 2,048 random directions:", best_cosines(W, noise).mean())
