"""The reference solutions against src/, and the samplers' and losses' own invariants. CPU, seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import pytest  # noqa: E402
import torch  # noqa: E402

import data as D  # noqa: E402
import diffusion as Df  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

V = 65


@pytest.fixture(scope="module")
def nets():
    torch.manual_seed(0)
    return {"mdm": GPT(GPTConfig(V, causal=False)).eval(), "ar": GPT(GPTConfig(V, causal=True)).eval()}


@pytest.fixture(scope="module")
def lines():
    _, train, _ = D.load()
    return D.batch(train, 8, torch.Generator().manual_seed(1))


# ------------------------------------------------------------- (a) the masked loss
def test_masked_loss_matches_src(nets, lines):
    import solution_a_masked_loss as sol

    m = nets["mdm"]
    t = torch.tensor([0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
    a = sol.mask_line(lines, t, m.cfg.mask_id, torch.Generator().manual_seed(3))
    b = Df.mask_line(lines, t, m.cfg.mask_id, torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    with torch.no_grad():
        mine = sol.masked_loss(m(a[0]), lines, a[1], t)
        src = Df.mdm_loss(m, lines, t, torch.Generator().manual_seed(3))
    assert torch.allclose(mine, src, rtol=1e-5)


def test_mask_rate_follows_t(lines):
    xt, masked = Df.mask_line(lines.repeat(50, 1), torch.full((400,), 0.3), V, torch.Generator().manual_seed(0))
    assert abs(masked.float().mean().item() - 0.3) < 0.01
    assert (xt[masked] == V).all() and (xt[~masked] == lines.repeat(50, 1)[~masked]).all()


# ------------------------------------------------------------- (b) unmasking
def test_reveal_counts_match_src():
    import solution_b_unmask as sol

    for n, s in ((10, 4), (128, 1), (128, 128), (128, 3), (16, 16), (72, 5)):
        assert sol.reveal_counts(n, s) == Df.reveal_counts(n, s)
        assert sum(Df.reveal_counts(n, s)) == n


def test_unmask_step_matches_src(nets, lines):
    import solution_b_unmask as sol

    m = nets["mdm"]
    x, _ = Df.mask_line(lines, torch.full((8,), 0.6), m.cfg.mask_id, torch.Generator().manual_seed(2))
    with torch.no_grad():
        logits = m(x)
    mine = sol.step(logits, x, m.cfg.mask_id, 5, "confidence")
    src, _ = Df.choose(logits, x, x == m.cfg.mask_id, 5, "confidence", 0, None)
    assert torch.equal(mine, src)
    l2r = sol.step(logits, x, m.cfg.mask_id, 3, "l2r")
    for row, before in zip(l2r, x):
        first = (before == m.cfg.mask_id).nonzero().flatten()[:3]
        assert (row[first] != m.cfg.mask_id).all() and ((row != before).sum() == 3)


def test_sampler_reveals_everything_once(nets):
    x, when = Df.sample_mdm(nets["mdm"], 4, 8, "random", 1.0, gen=torch.Generator().manual_seed(0), trace=True)
    assert (x < V).all() and when.min() == 1 and when.max() == 8
    assert all(int((when[0] == s).sum()) == 16 for s in range(1, 9))


# ------------------------------------------------------------- (c) block diffusion
def test_block_mask_matches_src():
    import solution_c_block_mask as sol

    for T, L in ((8, 2), (128, 4), (128, 16), (128, 128), (12, 1)):
        assert torch.equal(sol.block_mask(T, L), Df.block_mask(T, L))


def test_block_128_is_plain_masked_diffusion(nets, lines):
    m = nets["mdm"]
    xt, _ = Df.mask_line(lines, torch.full((8,), 0.5), m.cfg.mask_id, torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(Df.bd_forward(m, xt, lines, 128), m(xt), atol=1e-5)


def test_block_1_is_autoregressive(nets, lines):
    """With L = 1 a masked position sees only the clean characters before it: its logits equal a causal
    model's prediction from the previous position, when the weights are shared."""
    m = nets["mdm"]
    xt = torch.full_like(lines, m.cfg.mask_id)
    with torch.no_grad():
        bd = Df.bd_forward(m, xt, lines, 1)
        causal = GPT(GPTConfig(V, causal=True)).eval()
        causal.load_state_dict(m.state_dict())
        for i in (1, 7, 64, 127):
            seq = torch.cat([lines[:, :i], torch.full((8, 1), m.cfg.mask_id)], 1)
            assert torch.allclose(bd[:, i], causal(seq)[:, i], atol=1e-4)


@pytest.mark.parametrize("L", [4, 16])
def test_kv_cache_gives_the_same_characters(nets, L):
    a = Df.sample_bd(nets["mdm"], 3, L, L, gen=torch.Generator().manual_seed(1), cache=True)
    b = Df.sample_bd(nets["mdm"], 3, L, L, gen=torch.Generator().manual_seed(1), cache=False)
    assert torch.equal(a, b)


def test_ar_cache(nets):
    lu = torch.full((V,), -torch.log(torch.tensor(float(V))))
    a = Df.sample_ar(nets["ar"], 2, lu, T=40, gen=torch.Generator().manual_seed(0), cache=True)
    b = Df.sample_ar(nets["ar"], 2, lu, T=40, gen=torch.Generator().manual_seed(0), cache=False)
    assert torch.equal(a, b)


# ------------------------------------------------------------- scores
def test_untrained_scores_near_uniform(nets, lines):
    lu = torch.full((V,), -torch.log(torch.tensor(float(V))))
    for s in (Df.ar_nll(nets["ar"], lines, lu), Df.elbo(nets["mdm"], lines, K=8),
              Df.ordered_nll(nets["mdm"], lines[:2], "l2r")):
        assert abs(s - 4.174) < 0.4


def test_library_carry_over_and_llada():
    import library as Lb

    torch.manual_seed(0)
    m = GPT(GPTConfig(V, causal=False)).eval()
    _, _, val = D.load()
    r = Lb.check_loss(m, D.val_blocks(val, 16))
    assert r["max_relative_gap"] < 1e-5
    empty = torch.zeros((3, 0), dtype=torch.long)
    a = Lb.llada_generate(m, empty, steps=8, temperature=1.0, gen=torch.Generator().manual_seed(5))
    b = Df.sample_mdm(m, 3, 8, "confidence", 1.0, gen=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
