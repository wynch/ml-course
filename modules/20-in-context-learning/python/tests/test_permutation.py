"""The prompt-permutation test: break the pairing of x and y, and the net must lose its edge.

Runs on the CPU with the two committed nets (artifacts/), in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402  (MPS guard before torch)
import icl  # noqa: E402

CPU = torch.device("cpu")


def load(name: str) -> icl.ICLNet:
    net = icl.ICLNet()
    net.load_state_dict(torch.load(common.ARTIFACTS / f"{name}.pt"))
    return net.eval()


def mse(net, xs, ys):
    return icl.mse_over_d(icl.net_predict(net, xs, ys, CPU), ys)


def test_shuffled_pairs_give_the_variance_of_y():
    """Control net on fresh shuffled prompts: error about Var(y)/d = 1, the score of guessing 0."""
    xs, ys, _ = icl.eval_set(21, shuffled=True)
    assert 0.8 <= mse(load("icl_control"), xs, ys).mean() <= 1.2


def test_main_net_cannot_use_shuffled_pairs():
    """With y paired to the wrong x, the trained net does no better than guessing 0."""
    xs, ys, _ = icl.eval_set(22, shuffled=True)
    assert mse(load("icl_net"), xs, ys)[1:].mean() >= 0.8


def test_pair_order_barely_matters():
    """Least squares ignores the order of the pairs; reordering them moves the net's error little."""
    net = load("icl_net")
    xs, ys, _ = icl.eval_set(23, shuffled=False, n=500)
    rng = np.random.default_rng(0)
    perm = np.stack([np.r_[rng.permutation(20), 20] for _ in range(500)])
    xs_p = np.take_along_axis(xs, perm[..., None], 1)
    ys_p = np.take_along_axis(ys, perm, 1)
    a, b = mse(net, xs, ys)[20], mse(net, xs_p, ys_p)[20]
    assert b < 2 * a + 0.02


def test_no_peek_at_the_answer():
    """The guess for y_n must not change when y_n itself changes: the causal mask holds."""
    net = load("icl_net")
    xs, ys, _ = icl.eval_set(24, shuffled=False, n=64)
    x, y = torch.from_numpy(xs).float(), torch.from_numpy(ys).float()
    with torch.no_grad():
        p = net(x, y)
        for n in (0, 7, 20):
            y2 = y.clone()
            y2[:, n] += 100.0
            assert torch.allclose(net(x, y2)[:, n], p[:, n], atol=1e-5)
