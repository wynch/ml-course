"""The estimator check: TwoNN and MLE on sets whose dimension we know.

Runs on the CPU in about a second. If an estimator drifts more than 1 from the
true dimension on these sets, no intrinsic-dimension number in the module can be
trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402,F401  (MPS guard before torch)
import idest  # noqa: E402


@pytest.mark.parametrize("dim", [2, 5, 8])
def test_curved_cube(dim):
    e = idest.estimate(idest.curved_cube(dim, 4000, seed=100 + dim))
    assert abs(e["twonn"] - dim) < 1
    assert abs(e["mle"] - dim) < 1
    assert max(e["twonn"], e["mle"]) / min(e["twonn"], e["mle"]) < 1.3


def test_flat_subspace():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 4)) @ rng.normal(size=(4, 784))     # 4-d, linearly embedded
    e = idest.estimate(x)
    assert abs(e["twonn"] - 4) < 0.5 and abs(e["mle"] - 4) < 0.5


def test_duplicates_dropped():
    x = idest.curved_cube(3, 1000, seed=1)
    e = idest.estimate(np.concatenate([x, x[:50]]))
    assert e["duplicates_removed"] == 50 and e["n"] == 1000


def test_knn_exact():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 20))
    d = idest.knn_dists(x, 5, chunk=64)
    full = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    np.fill_diagonal(full, np.inf)
    assert np.allclose(d, np.sort(full, 1)[:, :5], atol=1e-9)
