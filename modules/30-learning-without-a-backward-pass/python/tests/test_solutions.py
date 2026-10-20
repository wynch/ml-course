"""The reference solutions, checked against src/ and autograd. CPU, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import torch  # noqa: E402

import rules as R  # noqa: E402

SMALL = (784, 64, 64, 64, 10)


def batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 784, generator=g), torch.randint(10, (n,), generator=g)


# ------------------------------------------------------------- (a) MeZO
def test_mezo_projected_grad_matches_src():
    import solution_a_mezo as sol

    x, y = batch()
    a, b = R.MeZO(SMALL, 0), R.MeZO(SMALL, 0)
    for seed in (1, 2, 3):
        mine = sol.projected_grad(lambda: b.loss(x, y), b.params, seed, 1e-3)
        ref, _ = a.projected_grad(x, y, seed)
        assert abs(mine - ref) < 1e-6
    assert all(torch.allclose(p, q) for p, q in zip(a.params, b.params))


def test_mezo_average_points_at_the_gradient():
    import solution_a_mezo as sol

    torch.manual_seed(0)
    A, w = torch.randn(20, 20), torch.randn(20)
    est = sol.estimate(lambda: float(0.5 * ((A @ w) ** 2).sum()), [w], 2000)[0]
    true = A.T @ (A @ w)
    assert float(torch.nn.functional.cosine_similarity(est, true, 0)) > 0.98


# --------------------------------------------------------- (b) Mono-Forward
def test_mono_forward_layer_matches_src():
    import solution_b_mono_forward as sol

    torch.manual_seed(0)
    w, b, m = torch.randn(16, 20) * 0.3, torch.randn(16) * 0.1, torch.randn(10, 16) * 0.3
    x, y = torch.rand(8, 20), torch.randint(10, (8,))
    for u, v in zip(sol.layer_grads(w, b, m, x, y), R.mf_layer_grads(w, b, m, x, y)):
        if isinstance(u, float):
            assert abs(u - v) < 1e-5
        else:
            assert torch.allclose(u, v, atol=1e-6)


# ------------------------------------------------------- (c) predictive coding
def test_pc_node_grads_match_src():
    import solution_c_pc as sol

    x, y = batch()
    pc = R.PredictiveCoding(SMALL, 0)
    xs = pc.infer(x, y, T=3)
    mine = sol.node_grads(pc.layers, xs, y)
    d = pc.deltas(xs, y)
    for l in range(1, len(pc.layers)):
        ref = -d[l] + (xs[l] > 0) * (d[l + 1] @ pc.layers[l][0])
        assert torch.allclose(mine[l - 1], ref, atol=1e-6)


def test_pc_infer_matches_src():
    import solution_c_pc as sol

    x, y = batch()
    pc = R.PredictiveCoding(SMALL, 0, gamma=0.1)
    for T in (0, 1, 5):
        a, b = sol.infer(pc.layers, x, y, T, 0.1), pc.infer(x, y, T=T)
        assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, b))
