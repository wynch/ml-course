"""The reference solutions and the hand rules, checked against the scripts. CPU, a few seconds, no model download.

Two tests run MLX on small random arrays (no model): the hand KV rule against
mx.quantize bit for bit, and the torch rule of exercise (c) against the MLX one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------- (a) bitlinear
def test_bitlinear_codes_match_script():
    import bitlinear
    import solution_a_bitlinear as sol

    W = torch.randn(48, 96, generator=torch.Generator().manual_seed(0)) * 0.03
    assert torch.equal(sol.ternary_codes(W), bitlinear.ternary_codes(W))
    assert set(torch.unique(sol.ternary_codes(W)).tolist()) <= {-1.0, 0.0, 1.0}
    W.requires_grad_(True)
    out = sol.ste_ternary(W)
    assert torch.allclose(out, bitlinear.ternary(W.detach()))
    out.sum().backward()
    assert torch.equal(W.grad, torch.ones_like(W))


def test_ties_go_to_zero():
    """A latent exactly on the threshold 0.5 * mean|W| gets code 0: torch rounds halves to even."""
    import bitlinear

    W = torch.tensor([[0.5, 1.5, -0.5, -1.5]])               # mean |W| = 1, so W / g = W
    assert bitlinear.ternary_codes(W).tolist() == [[0.0, 1.0, 0.0, -1.0]]
    assert bitlinear.ternary_codes(W.to(torch.bfloat16)).float().tolist() == [[0.0, 1.0, 0.0, -1.0]]


def test_bitlinear_forward():
    import bitlinear
    from tiny_gpt import new_gpt

    torch.manual_seed(0)
    m = bitlinear.to_bitnet(new_gpt(64, 2, vocab_size=20, block_size=16))
    lin = dict(bitlinear.block_linears(m))
    assert len(lin) == 8 and all(isinstance(v, bitlinear.BitLinear) for v in lin.values())
    x = torch.randn(3, 5, 64)
    mod = lin["blocks.0.attn.qkv"]
    want = torch.nn.functional.linear(bitlinear.act_quant(x), bitlinear.ternary(mod.weight), mod.bias)
    assert torch.allclose(mod(x), want, atol=1e-6)
    q = bitlinear.act_quant(x)
    s = 127 / x.abs().amax(-1, keepdim=True)
    assert torch.allclose(q * s, (q * s).round(), atol=1e-4)  # on the 8-bit grid


def test_resume_matches_unbroken(tmp_path):
    """A run paused at a deadline and resumed ends with the same weights as an unbroken one."""
    import time

    import train

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 20, 5000)
    for kind in ("float", "ternary"):
        a, _ = train.train(kind, 64, 1, 3e-3, 0, ids, 20, "cpu", steps=12)
        ck = tmp_path / f"{kind}.pt"
        none, _ = train.train(kind, 64, 1, 3e-3, 0, ids, 20, "cpu", steps=12, ckpt=ck, deadline=time.perf_counter())
        assert none is None and ck.exists()
        b, info = train.train(kind, 64, 1, 3e-3, 0, ids, 20, "cpu", steps=12, ckpt=ck, deadline=time.perf_counter() + 600)
        assert info["pieces"] == 2 and not ck.exists()
        for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(p, q), (kind, n)


# -------------------------------------------------------------- (b) bytes
def test_bytes_match_script():
    import matched
    import solution_b_bytes as sol
    from tiny_gpt import new_gpt

    for d, L in ((96, 2), (160, 5), (288, 6)):
        m = new_gpt(d, L, vocab_size=sol.VOCAB, block_size=sol.BLOCK)
        n_other = m.num_params() - 12 * d * d * L
        assert n_other == (sol.VOCAB + sol.BLOCK) * d + 13 * d * L + 2 * d
        assert abs(sol.model_bytes(d, L, 1.6, 0) + 8 * L - matched.arm_bytes("ternary", d, L, sol.VOCAB)) <= 1
        assert abs(sol.model_bytes(d, L, 4, 4 / 32) - matched.arm_bytes("4-bit", d, L, sol.VOCAB)) <= 1
        assert abs(sol.model_bytes(d, L, 2, 4 / 32) - matched.arm_bytes("2-bit", d, L, sol.VOCAB)) <= 1
        assert sol.model_bytes(d, L, 16, 0) == matched.arm_bytes("16-bit", d, L, sol.VOCAB) == 2 * m.num_params()


def test_shapes_fill_the_budget():
    import matched

    for budget in matched.BUDGETS:
        for arm in matched.ARMS:
            sh = matched.shapes(arm, budget, 317)
            assert len(set(sh)) == 3 and sh[0][1] > sh[1][1] > sh[2][1]
            for d, L in sh:
                assert abs(matched.arm_bytes(arm, d, L, 317) / budget - 1) <= matched.TOL
                assert d % 32 == 0


# ------------------------------------------------------------- (c) layout
def test_layout_solution():
    import solution_c_kv_layout as sol

    R = sol.hadamard(64)
    assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-6)
    K = sol.fake_keys()
    e = {(lay, rot is not None): sol.rel_error(K, sol.fq(K, 2, lay, rot)) for lay in ("token", "channel") for rot in (None, R)}
    assert e[("channel", False)] < e[("token", False)]
    assert e[("token", True)] < e[("token", False)]
    assert sol.rel_error(K, sol.fq(K, 8, "token")) < 0.02


def test_torch_rule_equals_mlx_rule():
    import mlx.core as mx
    import solution_c_kv_layout as sol

    from kvquant import fq_cache_tensor

    K = (sol.fake_keys(128, 64) * 0.1).to(torch.bfloat16).float()
    for bits in (2, 4):
        for layout in ("token", "channel"):
            mine = sol.fq(K, bits, layout).to(torch.bfloat16).float().numpy()
            theirs = np.array(fq_cache_tensor(mx.array(K.numpy())[None, None], bits, layout)[0, 0].astype(mx.float32))
            assert np.abs(mine - theirs).max() <= 1e-2 * np.abs(theirs).max(), (bits, layout)


def test_hand_kv_rule_is_mlx_bit_for_bit():
    import mlx.core as mx

    from kvquant import affine_fq, hadamard

    mx.random.seed(0)
    x = (mx.random.normal((2, 5, 96, 64)) * 0.5).astype(mx.bfloat16)
    x[0, 0, 0, :] = -mx.abs(x[0, 0, 0, :])                   # one group with no positive value
    for bits in (2, 3, 4, 8):
        q = mx.quantize(x, group_size=64, bits=bits)
        assert mx.array_equal(affine_fq(x, bits), mx.dequantize(*q, group_size=64, bits=bits)).item()
    R = hadamard(64)
    assert np.abs(R @ R.T - np.eye(64)).max() < 1e-6
