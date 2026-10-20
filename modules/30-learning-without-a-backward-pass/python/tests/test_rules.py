"""The six learners in src/rules.py against autograd and against each other. CPU, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402,F401  (laptop guard before torch)
import torch  # noqa: E402

import memory  # noqa: E402
import rules as R  # noqa: E402

SMALL = (784, 64, 64, 64, 10)


def batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 784, generator=g), torch.randint(10, (n,), generator=g)


def test_checkpointing_gives_the_same_weights_as_backprop():
    x, y = batch()
    a, b = R.Backprop(SMALL, 0), R.Backprop(SMALL, 0, segments=2)
    for _ in range(3):
        assert a.step(x, y) == b.step(x, y)
    assert all(torch.equal(p, q) for p, q in zip(a.params, b.params))


def test_hand_adam_matches_torch_adam():
    torch.manual_seed(0)
    p1 = torch.randn(5, 3)
    p2 = p1.clone().requires_grad_()
    mine, ref = R.Adam([p1], lr=1e-2), torch.optim.Adam([p2], lr=1e-2)
    for k in range(5):
        g = torch.randn(5, 3, generator=torch.Generator().manual_seed(k))
        mine.step([g])
        p2.grad = g.clone()
        ref.step()
    assert torch.allclose(p1, p2.detach(), atol=1e-6)


def test_mezo_restores_weights_and_moves_along_z():
    x, y = batch()
    m = R.MeZO(SMALL, 0)
    before = [p.clone() for p in m.params]
    g, _ = m.projected_grad(x, y, seed=123)
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(m.params, before))
    m._nudge(123, -m.lr * g)
    gen = torch.Generator().manual_seed(123)
    for p, q in zip(m.params, before):
        z = torch.randn(p.shape, generator=gen)
        assert torch.allclose(p - q, -m.lr * g * z, atol=1e-6)


def test_mezo_estimate_matches_the_directional_derivative():
    """With a small epsilon the two-point estimate is the derivative along z (at 1e-3, ReLU kinks crossed
    by the nudge already move it by a few per cent)."""
    x, y = batch()
    m = R.MeZO(SMALL, 0, eps=1e-4)
    ps = [p.clone().requires_grad_() for p in m.params]
    a = x
    for i in range(0, len(ps) - 2, 2):
        a = torch.relu(a @ ps[i].T + ps[i + 1])
    loss = torch.nn.functional.cross_entropy(a @ ps[-2].T + ps[-1], y)
    grads = torch.autograd.grad(loss, ps)
    gen = torch.Generator().manual_seed(7)
    exact = sum(float((gr * torch.randn(gr.shape, generator=gen)).sum()) for gr in grads)
    est, _ = m.projected_grad(x, y, seed=7)
    assert abs(est - exact) < 0.02 * abs(exact) + 1e-3


def test_ff_layer_grads_match_autograd():
    torch.manual_seed(0)
    w, b = torch.randn(16, 20) * 0.3, torch.randn(16) * 0.1
    xp, xn = torch.rand(8, 20), torch.rand(8, 20)
    loss, gw, gb, _, _ = R.ff_layer_grads(w, b, xp, xn, theta=0.5)
    w2, b2 = w.clone().requires_grad_(), b.clone().requires_grad_()
    gp = (torch.relu(xp @ w2.T + b2) ** 2).mean(1)
    gn = (torch.relu(xn @ w2.T + b2) ** 2).mean(1)
    L = torch.nn.functional.softplus(0.5 - gp).mean() + torch.nn.functional.softplus(gn - 0.5).mean()
    aw, ab = torch.autograd.grad(L, [w2, b2])
    assert abs(loss - float(L.detach())) < 1e-5
    assert torch.allclose(gw, aw, atol=1e-5) and torch.allclose(gb, ab, atol=1e-5)


def test_mf_layer_grads_match_autograd_and_stop_at_the_input():
    torch.manual_seed(0)
    w, b, m = torch.randn(16, 20) * 0.3, torch.randn(16) * 0.1, torch.randn(10, 16) * 0.3
    x, y = torch.rand(8, 20, requires_grad=True), torch.randint(10, (8,))
    loss, gw, gb, gm, _ = R.mf_layer_grads(w, b, m, x.detach(), y)
    w2, b2, m2 = (t.clone().requires_grad_() for t in (w, b, m))
    L = torch.nn.functional.cross_entropy(torch.relu(x.detach() @ w2.T + b2) @ m2.T, y)
    aw, ab, am = torch.autograd.grad(L, [w2, b2, m2])
    assert abs(loss - float(L.detach())) < 1e-5
    assert torch.allclose(gw, aw, atol=1e-5) and torch.allclose(gb, ab, atol=1e-5) and torch.allclose(gm, am, atol=1e-5)


def _energy_autograd(pc, xs, y):
    xs2 = [t.clone().requires_grad_() for t in xs]
    ws = [t.clone().requires_grad_() for wb in pc.layers for t in wb]
    L = len(pc.layers)

    def mu(l, xp):
        return pc.phi(l - 1, xp) @ ws[2 * (l - 1)].T + ws[2 * (l - 1) + 1]

    F = sum(0.5 * ((xs2[l] - mu(l, xs2[l - 1])) ** 2).sum() for l in range(1, L))
    F = F + torch.nn.functional.cross_entropy(mu(L, xs2[L - 1]), y, reduction="sum")
    g = torch.autograd.grad(F, ws + xs2[1:])
    return float(F.detach()), g[:len(ws)], g[len(ws):]


def test_pc_gradients_match_autograd_of_the_energy():
    x, y = batch()
    pc = R.PredictiveCoding(SMALL, 0)
    xs = pc.infer(x, y, T=4)
    F, gw, gx = _energy_autograd(pc, xs, y)
    assert abs(pc.energy(xs, y) - F) < 1e-3 * abs(F)
    mine = pc.weight_grads(xs, y)
    assert all(torch.allclose(u * len(y), v, atol=1e-4) for u, v in zip(mine, gw))
    d = pc.deltas(xs, y)
    for l in range(1, len(pc.layers)):
        node = -d[l] + (xs[l] > 0) * (d[l + 1] @ pc.layers[l][0])
        assert torch.allclose(node, gx[l - 1], atol=1e-5)


def test_pc_matches_backprop_on_arrival():
    x, y = batch()
    bp = R.Backprop(SMALL, 0)
    ref = torch.autograd.grad(torch.nn.functional.cross_entropy(bp.forward(x), y), bp.params)
    pc = R.PredictiveCoding(SMALL, 0)
    for T, layer in ((0, 3), (1, 2), (2, 1), (3, 0)):        # the error reaches one more layer per step
        g = pc.weight_grads(pc.infer(x, y, T=T), y)
        cos = torch.nn.functional.cosine_similarity(g[2 * layer].flatten(), ref[2 * layer].flatten(), 0)
        assert cos > 0.999
        if layer > 0:
            assert float(g[2 * (layer - 1)].abs().max()) == 0.0


def test_pc_feedforward_is_the_backprop_network():
    x, _ = batch()
    assert torch.allclose(R.PredictiveCoding(SMALL, 0).predict(x), R.Backprop(SMALL, 0).predict(x), atol=1e-5)


def test_forward_forward_prediction_uses_the_overlay():
    x, y = batch()
    o = R.overlay(x, y)
    assert torch.equal(o[:, 10:], x[:, 10:]) and torch.equal(o[:, :10].argmax(1), y) and float(o[:, :10].sum()) == len(y)


def test_live_bytes_meter():
    meter = memory.LiveBytes()
    with meter:
        a = torch.zeros(1024, 256)                     # 1 MiB
        b = a + 1                                      # another
        del a
        c = b[:10]                                     # a view: no new storage
        b2 = c * 2                                     # 10 KiB
    assert meter.peak == 2 * 2**20
    assert meter.live == 2**20 + 10 * 256 * 4
    del b, c, b2


def test_sgd_keeps_no_state():
    p = torch.ones(3)
    R.SGD([p], lr=0.5).step([torch.ones(3)])
    assert torch.equal(p, torch.full((3,), 0.5))
