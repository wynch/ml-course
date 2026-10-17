"""The reference solutions against the module's code, and the models' two forms.

Everything here runs on the CPU with small random models, in a few seconds.
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

import models  # noqa: E402


# ----------------------------------------------------------------- (a) GRU
def test_gru_cell_matches_nn_gru():
    import solution_a_gru_cell as sol

    torch.manual_seed(0)
    gru = torch.nn.GRU(16, 32, batch_first=True)
    x = torch.randn(3, 40, 16)
    with torch.no_grad():
        ref, _ = gru(x)
        assert torch.allclose(sol.run_gru(x, gru), ref, atol=1e-5)


def test_gru_cell_matches_module_gru_layer():
    import solution_a_gru_cell as sol

    torch.manual_seed(1)
    m = models.GRULM(65, hidden=24, n_layer=1).eval()
    x = m.emb(torch.randint(0, 65, (2, 30)))
    with torch.no_grad():
        ref, _ = m.rnn(x)
        assert torch.allclose(sol.run_gru(x, m.rnn), ref, atol=1e-5)


# ---------------------------------------------------------------- (b) scan
def test_selective_loop_matches_matrix_form():
    import solution_b_scan as sol

    torch.manual_seed(0)
    block = models.SelectiveBlock(32, n_head=4)
    xs = torch.randn(2, 64, block.E)
    extra = torch.randn(2, 64, 2 * block.N + block.H)
    with torch.no_grad():
        y, S = sol.selective_loop(block, xs, extra)
        assert torch.allclose(y, block.core(xs, extra), atol=1e-4)
    assert S.shape == (2, 4, block.P, block.N)


# -------------------------------------------------------------- (c) kernel
def test_lti_kernel_and_conv_match_block():
    import solution_c_lti_kernel as sol

    torch.manual_seed(0)
    block = models.LTIBlock(16)
    T = 48
    with torch.no_grad():
        dt, a = block._decay()
        K = sol.kernel(a, block.B, block.C, dt, T)
        assert torch.allclose(K, block.kernel(T), atol=1e-5)
        xs = torch.randn(2, T, block.E)
        assert torch.allclose(sol.causal_conv(xs, K), block.core(xs, None), atol=1e-4)
        assert torch.allclose(sol.recurrence(xs, a, block.B, block.C, dt), block.core(xs, None), atol=1e-4)


# ------------------------------------------------- the models' two forms
@pytest.mark.parametrize("kind", ["gru", "lti", "selective", "gpt"])
def test_step_equals_parallel_forward(kind):
    torch.manual_seed(0)
    m = models.build(kind, 65).eval()
    s = models.GPTCache(m) if kind == "gpt" else m
    idx = torch.randint(0, 65, (2, 40))
    with torch.no_grad():
        full, _ = m(idx)
        state = s.init_state(2, "cpu")
        outs = []
        for t in range(idx.shape[1]):
            lo, state = s.step(idx[:, t], state)
            outs.append(lo)
    assert torch.allclose(torch.stack(outs, 1), full, atol=1e-4)


@pytest.mark.parametrize("kind", ["gru", "lti", "selective", "gpt"])
def test_parallel_forward_is_causal(kind):
    """In training mode, changing the tokens from t on leaves the logits before t untouched."""
    torch.manual_seed(0)
    m = models.build(kind, 65).train()
    x = torch.randint(0, 65, (4, 128))
    for t in (1, 17, 64, 127):
        x2 = x.clone()
        x2[:, t:] = torch.randint(0, 65, (4, 128 - t))
        a, _ = m(x)
        b, _ = m(x2)
        assert (a[:, :t] - b[:, :t]).abs().max() < 1e-5
        assert (a[:, t:] - b[:, t:]).abs().max() > 1e-3


def test_targets_are_next_characters():
    data = common.CharData()
    x, y = data.batch("train", torch.Generator().manual_seed(0), "cpu", batch=4, block=32)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_sizes_match_the_tiny_gpt():
    gpt = models.n_params(models.build("gpt", 65))
    assert gpt == 619776
    for kind in ("gru", "lti", "selective"):
        assert abs(models.n_params(models.build(kind, 65)) / gpt - 1) < 0.03


def test_state_sizes():
    gru = models.build("gru", 65)
    assert gru.state_floats(gru.init_state(1, "cpu")) == 2 * 224
    sel = models.build("selective", 65)
    # per layer: conv buffer E x 3, state H x P x N = E x N
    assert sel.state_floats(sel.init_state(1, "cpu")) == 5 * (280 * 3 + 280 * 16)
