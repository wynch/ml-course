"""The reference solutions, checked against the scripts. All run on the CPU in a few seconds.

One test runs mx.quantize on a small random matrix (no model) to check the
hand rule in src/compress.py bit for bit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import torch  # noqa: E402


# ------------------------------------------------------------ (a) distil
def test_distil_loss_matches_script():
    import distil
    import solution_a_distil_loss as sol

    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(2, 5, 65, generator=g), torch.randn(2, 5, 65, generator=g)
    y = torch.randint(65, (2, 5), generator=g)
    for T, a in ((1.0, 0.5), (2.0, 0.5), (4.0, 1.0), (2.0, 0.0)):
        assert torch.allclose(sol.distil_loss(s, t, y, T, a), distil.distil_loss(s, t, y, T, a), atol=1e-5)
    assert float(sol.distil_loss(t, t, y, 2.0, 1.0)) < 1e-5                 # a perfect copy costs nothing


def test_soften_flattens():
    import solution_a_distil_loss as sol

    x = torch.tensor([3.0, 1.0, 0.0])
    assert sol.soften(x, 1.0)[0] > sol.soften(x, 2.0)[0] > sol.soften(x, 8.0)[0] > 1 / 3


# ------------------------------------------------------------- (b) prune
def test_prune_mask_matches_script():
    import compress
    import solution_b_prune_tiny as sol

    W = torch.randn(64, 96, generator=torch.Generator().manual_seed(1))
    for f in (0.0, 0.1, 0.3, 0.5):
        mine = sol.magnitude_mask(W, f)
        assert torch.equal(mine, compress.magnitude_mask(W, f))
        assert int((~mine).sum()) == round(f * W.numel())
        if 0 < f:
            assert W[~mine].abs().max() <= W[mine].abs().min()


# ------------------------------------------------------------- (c) quant
def test_minmax_quant():
    import compress
    import solution_c_group_quant as sol

    W = torch.randn(32, 128, generator=torch.Generator().manual_seed(2))
    for bits in (3, 4):
        for g in (16, 64, 128):
            W_hat, bpw = sol.quantize_minmax(W, bits, g)
            assert bpw == bits + 32 / g
            Wg, Hg = W.view(32, -1, g), W_hat.view(32, -1, g)
            step = (Wg.amax(-1) - Wg.amin(-1)) / (2**bits - 1)
            assert ((Wg - Hg).abs().amax(-1) <= step / 2 + 1e-5).all()          # never off by more than half a step
            assert all(len(torch.unique(row)) <= 2**bits for row in Hg.reshape(-1, g))
    mine = sol.rel_error(W, sol.quantize_minmax(W, 4, 64)[0])
    theirs = sol.rel_error(W, compress.quantize_affine(W, 4, 64)[0])
    assert abs(mine - theirs) / theirs < 0.2


def test_hand_rule_is_mlx_bit_for_bit():
    import mlx.core as mx
    import numpy as np

    import compress

    W = (torch.randn(48, 256, generator=torch.Generator().manual_seed(3)) * 0.05).to(torch.bfloat16)
    W[0, :64] = -W[0, :64].abs()                                            # one group with no positive weight
    q, s, b = mx.quantize(mx.array(W.float().numpy()).astype(mx.bfloat16), group_size=64, bits=4)
    theirs = np.array(mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32))
    mine = compress.quantize_affine(W.float(), 4, 64)[0].to(torch.bfloat16).float().numpy()
    assert np.array_equal(mine, theirs)
