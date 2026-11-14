"""The reference solutions, checked against the scripts and the committed results. CPU, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)


# ------------------------------------------------------------ (a) count
@pytest.mark.parametrize("d", common.SIZES)
def test_count_matches_model(d):
    import solution_a_count as sol
    from char_gpt import GPT, GPTConfig

    m = GPT(GPTConfig(vocab_size=142, block_size=128, n_layer=4, n_head=4, d_model=d, dropout=0.0))
    assert sol.non_embedding_params(d) == m.num_params_non_embedding()


def test_count_matches_results():
    import solution_a_count as sol

    for d, r in common.read_runs().items():
        run = r["runs"][0]
        assert sol.non_embedding_params(d) == run["n_non_embedding"]
        assert run["curve"][-1]["tokens"] == sol.TOKENS
    assert sol.compute(1_000_000, 1_000) == 6e9


# ------------------------------------------------------------ (b) fit
def test_fit_matches_script():
    import fit
    import solution_b_fit as sol

    r = common.read_results("fit")
    alpha, a, r2 = sol.fit_power(r["n_non_embedding"], r["loss_4M_seed0"])
    ref = fit.fit_power(r["n_non_embedding"], r["loss_4M_seed0"])
    assert alpha == pytest.approx(ref["exponent"]) and r2 == pytest.approx(ref["r2"])
    assert alpha == pytest.approx(r["l_of_n_main"]["exponent"])


def test_fit_recovers_known_alpha():
    import solution_b_fit as sol

    alpha, _, r2 = sol.fit_power(sol.N, sol.synthetic(0.1, 0.0, 0))
    assert alpha == pytest.approx(0.1) and r2 == pytest.approx(1.0)
    assert 0.08 <= sol.fit_power(sol.N, sol.synthetic(0.1, 0.01, 0))[0] <= 0.12


# ------------------------------------------------------------ (c) frontier
def test_frontier_matches_script():
    import solution_c_frontier as sol

    fr = common.read_results("frontier")
    runs = sol.load_seed0()
    for row in fr["rows"]:
        d, loss = sol.best_size(runs, row["C"])
        assert d == row["best_d"]
        assert loss == pytest.approx(row["loss"])
    assert sol.loss_at(runs[32], 1e9) is None


# ------------------------------------------------------------ (d) router
def test_router_matches_script():
    import moe
    import solution_d_router as sol
    import torch

    torch.manual_seed(0)
    logits = torch.randn(512, 8)
    p1, w1, i1 = sol.route(logits, 2)
    p2, w2, i2 = moe.route(logits, 2)
    assert torch.equal(i1, i2) and torch.allclose(w1, w2) and torch.allclose(w1.sum(-1), torch.ones(512))
    assert float(sol.balance_loss(p1, i1)) == pytest.approx(float(moe.balance_loss(p2, i2)))
    # uniform routing scores exactly 1
    ids = torch.arange(512 * 2).reshape(512, 2) % 8
    assert float(sol.balance_loss(torch.full((512, 8), 1 / 8), ids)) == pytest.approx(1.0)


def test_router_mixtral_record():
    import solution_d_router as sol

    m = sol.mixtral_active()
    assert m["active"] == m["record_active"]


def test_moe_gate_equals_dispatch():
    """Training runs every expert and zeroes the unchosen ones; the dispatch runs each on its own tokens."""
    import moe
    import torch

    torch.manual_seed(0)
    for spec in (dict(experts=8, k=2, hidden=32), dict(experts=12, k=3, hidden=16, shared_hidden=24)):
        layer = moe.MoE(16, n_layer=4, **spec)
        x = torch.randn(3, 7, 16, requires_grad=True)
        y1 = layer(x)
        g1 = torch.autograd.grad(y1.square().sum(), [x, layer.experts.w1, layer.router.weight])
        layer.sparse = True
        y2 = layer(x)
        g2 = torch.autograd.grad(y2.square().sum(), [x, layer.experts.w1, layer.router.weight])
        assert torch.allclose(y1, y2, atol=1e-6)
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2))


def test_moe_arm_sizes():
    """MoE arms: active size of dense, total size of dense_wide (within 3%)."""
    import moe

    c = {a: moe.counts(moe.build(a, 142, 0), a) for a in moe.ARMS}
    for a in ("moe8", "moe8_bal", "moe32_bal", "moe30s_bal"):
        assert abs(c[a]["n_active"] / c["dense"]["n_non_embedding"] - 1) < 0.03
        assert abs(c[a]["n_non_embedding"] / c["dense_wide"]["n_non_embedding"] - 1) < 0.03
