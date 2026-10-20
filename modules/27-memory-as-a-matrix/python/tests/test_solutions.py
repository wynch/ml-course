"""The reference solutions against the module's code, and the module's two forms against each other.

Everything here runs on the CPU with small random inputs, in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (MPS guard before torch)
import torch  # noqa: E402

import memory  # noqa: E402
import models  # noqa: E402
from m18_selective import selective_matrix_form  # noqa: E402


def _inputs(T=128, B=2, H=3, dk=16, dv=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = memory.unit(torch.randn(B, H, T, dk, generator=g, dtype=torch.float64))
    k = memory.unit(torch.randn(B, H, T, dk, generator=g, dtype=torch.float64))
    v = torch.randn(B, H, T, dv, generator=g, dtype=torch.float64)
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    la = -0.3 * torch.rand(B, H, T, generator=g, dtype=torch.float64)
    return q, k, v, beta, la


# ------------------------------------------------------------- (a) delta step
def test_delta_step_solution():
    import solution_a_delta_step as sol

    torch.manual_seed(0)
    S = torch.randn(8, 6, dtype=torch.float64)
    k, v = memory.unit(torch.randn(6, dtype=torch.float64)), torch.randn(8, dtype=torch.float64)
    assert torch.allclose(sol.delta_write(S, k, v, 0.4), sol.sgd_step(S, k, v, 0.4), atol=1e-12)
    assert torch.allclose(sol.linear_write(S, k, v, 0.4), S + 0.4 * torch.outer(v, k))
    # the module's write_read uses the same rules
    keys, vals = memory.unit(torch.randn(5, 6, dtype=torch.float64)), torch.randn(5, 8, dtype=torch.float64)
    for rule, write in (("linear", sol.linear_write), ("delta", sol.delta_write)):
        S = torch.zeros(8, 6, dtype=torch.float64)
        for kk, vv in zip(keys, vals):
            S = write(S, kk, vv)
        assert torch.allclose(keys @ S.T, memory.write_read(keys, vals, rule))


def test_delta_overwrites_linear_blends():
    keys = torch.eye(4, dtype=torch.float64)[[0, 1, 0]]
    vals = torch.tensor([[1.0, 0], [0, 1.0], [0, -1.0]], dtype=torch.float64)
    assert torch.allclose(memory.write_read(keys, vals, "delta")[2], vals[2])
    assert torch.allclose(memory.write_read(keys, vals, "linear")[2], vals[0] + vals[2])


# ------------------------------------------------------------- (b) one chunk
def test_chunk_solution():
    import solution_b_chunk as sol

    q, k, v, beta, _ = _inputs(T=32)
    loop, _ = memory.recurrent(q, k, v, beta, torch.zeros_like(beta), delta=True)
    assert torch.allclose(sol.delta_chunk(q, k, v, beta), loop, atol=1e-10)
    A = (beta[..., None] * (k @ k.transpose(-1, -2))).tril(-1)
    R = torch.randn(*A.shape[:-1], 5, dtype=torch.float64)
    X = sol.forward_substitution(A, R)
    eye = torch.eye(A.shape[-1], dtype=torch.float64)
    assert torch.allclose((eye + A) @ X, R, atol=1e-10)
    assert torch.allclose(memory.unit_lower_inverse(A) @ R, X, atol=1e-10)


# ------------------------------------------------------- (c) cache bytes
def test_cache_solution():
    import solution_c_cache as sol

    lib = common.read_results("library")
    for key, p in lib["models"].items():
        for T, row in p["contexts"].items():
            assert sol.cache_bytes(p["config"], p["layer_types"], int(T)) == row["cache_bytes"], (key, T)


# ------------------------------------------------ the two forms of every rule
@pytest.mark.parametrize("rule", list(memory.RULES))
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_equals_recurrent(rule, chunk):
    delta, gated = memory.RULES[rule]
    q, k, v, beta, la = _inputs()
    la = la if gated else torch.zeros_like(la)
    o1, S1 = memory.recurrent(q, k, v, beta, la, delta)
    o2, S2 = memory.chunked(q, k, v, beta, la, delta, chunk=chunk)
    assert torch.allclose(o1, o2, atol=1e-9) and torch.allclose(S1, S2, atol=1e-9)


def test_chunked_carries_a_state_in():
    q, k, v, beta, la = _inputs(T=64)
    S0 = torch.randn(2, 3, 24, 16, dtype=torch.float64)
    o1, _ = memory.recurrent(q, k, v, beta, la, True, S0=S0)
    o2, _ = memory.chunked(q, k, v, beta, la, True, chunk=32, S0=S0)
    assert torch.allclose(o1, o2, atol=1e-9)


def test_decay_rule_is_module_18_selective_recurrence():
    g = torch.Generator().manual_seed(1)
    B, T, H, P, N = 2, 48, 3, 5, 8
    xs = torch.randn(B, T, H, P, generator=g, dtype=torch.float64)
    Bm, Cm = torch.randn(B, T, N, generator=g, dtype=torch.float64), torch.randn(B, T, N, generator=g, dtype=torch.float64)
    dt = torch.rand(B, T, H, generator=g, dtype=torch.float64)
    log_a = -2 * dt
    ref = selective_matrix_form(xs, Bm, Cm, dt, log_a)
    o, _ = memory.chunked(Cm[:, None].expand(B, H, T, N), Bm[:, None].expand(B, H, T, N), xs.transpose(1, 2),
                          dt.transpose(1, 2), log_a.transpose(1, 2), delta=False, chunk=16)
    assert torch.allclose(o.transpose(1, 2), ref, atol=1e-10)


def test_unit_lower_inverse():
    g = torch.Generator().manual_seed(2)
    for C in (2, 4, 16, 64):
        A = torch.randn(3, C, C, generator=g, dtype=torch.float64).tril(-1) * 0.3
        eye = torch.eye(C, dtype=torch.float64)
        assert torch.allclose(memory.unit_lower_inverse(A) @ (eye + A), eye, atol=1e-10)


def test_inverse_and_chunks_hold_in_float32_when_keys_are_alike():
    """Keys nearly equal and write strengths near 1: the case that broke the first run's product formula."""
    g = torch.Generator().manual_seed(3)
    B, H, T, d = 2, 4, 128, 16
    base = torch.randn(B, H, 1, d, generator=g)
    k = memory.unit(base + 0.02 * torch.randn(B, H, T, d, generator=g))
    q = memory.unit(base + 0.02 * torch.randn(B, H, T, d, generator=g))
    v = torch.randn(B, H, T, d, generator=g)
    beta = 0.9 + 0.1 * torch.rand(B, H, T, generator=g)
    A = (beta[..., :64, None] * (k[:, :, :64] @ k[:, :, :64].transpose(-1, -2))).tril(-1)
    eye = torch.eye(64)
    ref = torch.linalg.solve_triangular((eye + A).double(), eye.double().expand(B, H, 64, 64), upper=False,
                                        unitriangular=True)
    assert (memory.unit_lower_inverse(A).double() - ref).abs().max() < 1e-4
    for rule in ("delta", "gated_delta"):
        la = -0.01 * torch.rand(B, H, T, generator=g) if rule == "gated_delta" else torch.zeros(B, H, T)
        o1, _ = memory.recurrent(q, k, v, beta, la, True)
        o2, _ = memory.chunked(q, k, v, beta, la, True, chunk=64)
        assert (o1 - o2).abs().max() < 1e-3


# ----------------------------------------------------------------- models
@pytest.mark.parametrize("rule", ["linear", "decay", "delta", "gated_delta"])
def test_model_chunked_equals_loop(rule):
    torch.manual_seed(0)
    m = models.RecallLM(64, 32, 8, [rule, rule], chunk=16).eval()
    x = torch.randint(0, 64, (2, 48))
    with torch.no_grad():
        assert torch.allclose(m(x), m(x, recurrent=True), atol=1e-4)


@pytest.mark.parametrize("layers", [["gated_delta"] * 2, ["attention"] * 2, ["delta", "attention"]])
def test_models_are_causal(layers):
    torch.manual_seed(0)
    m = models.RecallLM(64, 32, 8, layers, chunk=16)
    x = torch.randint(0, 64, (2, 64))
    for t in (1, 20, 63):
        x2 = x.clone()
        x2[:, t:] = torch.randint(0, 64, (2, 64 - t))
        a, b = m(x), m(x2)
        assert (a[:, :t] - b[:, :t]).abs().max() < 1e-5
        assert (a[:, t:] - b[:, t:]).abs().max() > 1e-4


def test_state_sizes():
    m = models.RecallLM(512, 128, 32, ["gated_delta", "gated_delta"])
    assert m.state_floats(256) == 2 * 4 * 32 * 32                      # heads x d_k x d_v, per layer
    a = models.RecallLM(512, 128, 32, ["attention", "attention"])
    assert a.state_floats(256) == 2 * 2 * 256 * 128                    # keys and values, every token


def test_mqar_batches():
    data = models.MQAR(512, 256)
    x, y = data.batch(16, 4, torch.Generator().manual_seed(0))
    for b in range(4):
        keys, vals = x[b, 0:32:2], x[b, 1:32:2]
        assert len(set(keys.tolist())) == 16 and (keys < 256).all() and (vals >= 256).all()
        table = dict(zip(keys.tolist(), vals.tolist()))
        assert (y[b, :32] == -100).all()
        for t in range(32, 256):
            assert table[x[b, t].item()] == y[b, t].item()
