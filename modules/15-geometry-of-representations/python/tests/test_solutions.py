"""The reference solutions, checked against the scripts. CPU only, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import idest  # noqa: E402


# ------------------------------------------------------------------ (a) id
def test_id_matches_script():
    import solution_a_id as sol

    d = idest.knn_dists(idest.curved_cube(6, 2000, seed=3), 20)
    assert abs(sol.twonn(d) - idest.twonn(d)) < 1e-9
    for k in (10, 15, 20):
        assert abs(sol.mle_k(d, k) - idest.mle_k(d, k)) < 1e-9


# ------------------------------------------------------------------ (b) regions
def test_regions_matches_script_and_sampling():
    import regions
    import solution_b_regions as sol

    rng = np.random.default_rng(0)
    for w in (16, 64):
        weights = regions.net_weights(regions.fashion_net(w, seed=w))
        for _ in range(3):
            a, b = rng.random(784), rng.random(784)
            exact = sol.count_on_segment(weights, a, b)
            assert exact == regions.count_on_segment(weights, a, b)[1]
            t = np.linspace(0, 1, 100_001)[:, None]
            ha, hb = a @ weights[0][0] + weights[0][1], b @ weights[0][0] + weights[0][1]
            h1 = (1 - t) * ha + t * hb          # layer 1 is affine in t; skips a 0.6 GB (100,001 x 784) array
            h2 = np.maximum(h1, 0) @ weights[1][0] + weights[1][1]
            pat = np.concatenate([h1 > 0, h2 > 0], 1)
            assert exact == (pat[1:] != pat[:-1]).any(1).sum() + 1


def test_tiling_covers_the_square():
    import regions

    weights = regions.net_weights(regions.toy_net(8, seed=3))
    cells = regions.tile(weights, box=2.0)
    assert abs(sum(regions.area(c) for c in cells) - 16.0) < 1e-9
    g = np.linspace(-2, 2, 1500)
    xx, yy = np.meshgrid(g, g)
    x = np.c_[xx.ravel(), yy.ravel()]
    h1 = x @ weights[0][0] + weights[0][1]
    h2 = np.maximum(h1, 0) @ weights[1][0] + weights[1][1]
    seen = len(np.unique(np.concatenate([h1 > 0, h2 > 0], 1), axis=0))
    assert seen <= len(cells) and seen >= 0.95 * len(cells)   # a grid can miss slivers, never add


# ------------------------------------------------------------------ (c) cka
def test_cka_matches_script_and_invariances():
    import cka
    import solution_c_cka as sol

    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 30))
    y = np.maximum(x @ rng.normal(size=(30, 40)), 0)
    assert abs(sol.linear_cka(x, y) - cka.linear_cka(x, y)) < 1e-12
    q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    assert abs(sol.linear_cka(x, 4 * x @ q + 2) - 1) < 1e-9
    assert abs(sol.linear_cka(x, y) - sol.linear_cka(y, x)) < 1e-12
    assert sol.linear_cka(x, rng.normal(size=(500, 30))) < 0.15
