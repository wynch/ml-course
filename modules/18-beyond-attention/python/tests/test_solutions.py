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


# ----------------------------------------------------------------- (d) window
def test_window_solution_matches_script():
    import solution_d_window as sol
    import window

    for T, w in ((8, 3), (40, 8), (40, None), (40, 64)):
        assert torch.equal(sol.window_mask(T, w), window.window_mask(T, w))
    for ws in window.ARMS.values():
        assert sol.reach(ws) == window.reach(ws)
    assert window.reach([16, 16, 16]) < window.copy_distance() < window.reach([32, 32, 32])


def test_window_kv_share_from_records():
    import json

    import solution_d_window as sol

    gemma = json.loads((sol.RECORDS / "gemma-3-27b.json").read_text())
    ink = json.loads((sol.RECORDS / "inkling-2-small.json").read_text())
    g, i = sol.kv_share(gemma, 131072), sol.kv_share(ink, 131072)
    assert (g["local"], g["global"], g["window"]) == (52, 10, 1024)
    assert (i["local"], i["global"], i["window"]) == (40, 8, 1024)
    assert g["share"] == pytest.approx((52 * 1024 + 10 * 131072) / (62 * 131072))
    assert sol.kv_share(gemma, 512)["share"] == 1.0          # shorter than the window: nothing saved


def test_window_mask_hides_older_tokens():
    """Changing a token more than reach() positions back leaves the last logit unchanged."""
    import window

    torch.manual_seed(0)
    g = window.make([8, 8, 8], 65, 0).eval()
    x = torch.randint(0, 65, (1, 64))
    y = x.clone()
    y[0, 63 - 22] = (y[0, 63 - 22] + 1) % 65                 # 22 back: beyond the reach of 21
    with torch.no_grad():
        assert torch.equal(g(x)[0][0, -1], g(y)[0][0, -1])
        z = x.clone()
        z[0, 63 - 21] = (z[0, 63 - 21] + 1) % 65             # 21 back: inside it
        assert not torch.equal(g(x)[0][0, -1], g(z)[0][0, -1])


def test_no_arm_sees_the_future_and_one_window_includes_itself():
    """Every arm: a change after position t leaves logits up to t unchanged.
    One layer of window w: a change w - 1 back reaches the current token, w back does not."""
    import window

    torch.manual_seed(0)
    x = torch.randint(0, 65, (1, 128))
    with torch.no_grad():
        for ws in window.ARMS.values():
            g = window.make(ws, 65, 0).eval()
            for p in g.parameters():
                p.data.normal_(0, 0.2)
            base = g(x)[0][0]
            for t in (0, 31, 100):
                y = x.clone()
                y[0, t + 1:] = (y[0, t + 1:] + 7) % 65
                assert torch.equal(g(y)[0][0, : t + 1], base[: t + 1])
        for w in (8, 64):
            g = window.make([w, None, None], 65, 0).eval()
            g.blocks = g.blocks[:1]
            for p in g.parameters():
                p.data.normal_(0, 0.2)
            out = g(x)[0][0, 120]
            for back, changes in ((w - 1, True), (w, False)):
                y = x.clone()
                y[0, 120 - back] = (y[0, 120 - back] + 1) % 65
                assert torch.equal(g(y)[0][0, 120], out) is not changes


def test_banded_equals_window_mask():
    import window
    import window_cost

    torch.manual_seed(0)
    a = models.make_gpt(65, block_size=256)
    for p in a.parameters():
        p.data.normal_(0, 0.3)
    b = models.make_gpt(65, block_size=256)
    b.load_state_dict(a.state_dict())
    window.set_windows(a, [64] * 3)
    window_cost.make_banded(b, 64)
    x = torch.randint(0, 65, (2, 256))
    assert torch.allclose(a(x)[0], b(x)[0], atol=1e-5)


@pytest.mark.parametrize("windows", [[16, 16, 16], [16, 16, None]])
def test_window_cache_matches_parallel(windows):
    import window
    import window_cost

    torch.manual_seed(0)
    g = window.make(windows, 65, 0).eval()
    s = window_cost.WindowCache(g, windows)
    x = torch.randint(0, 65, (1, 100))
    st, outs = s.init_state(1, "cpu"), []
    for t in range(100):
        logits, st = s.step(x[0, t:t + 1], st)
        outs.append(logits)
    assert torch.allclose(torch.cat(outs), g(x)[0][0], atol=1e-4)
    full = sum(w is None for w in windows)
    assert s.state_floats(st) == 2 * 128 * ((3 - full) * 16 + full * 100)
