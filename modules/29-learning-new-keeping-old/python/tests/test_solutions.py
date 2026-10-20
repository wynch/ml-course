"""The data, LoRA, the measures and the reference solutions, against src/. CPU only, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data  # noqa: E402
import forget  # noqa: E402
import lora  # noqa: E402
import plasticity  # noqa: E402


# ------------------------------------------------------------------ data
def test_texts_and_vocab():
    t = data.texts()
    assert t["a_train"].startswith("First Citizen:")
    assert t["b_train"].startswith("Q: Natalia sold clips")
    assert "####" in t["b_val"]
    v = data.vocab()
    assert len(v) == len(set(v)) == 91
    assert all(c == "\n" or 32 <= ord(c) < 127 for c in v)
    assert data.clean("Janet’s 3×4 – x") == "Janet's 3*4 - x"
    assert data.decode(data.encode("ROMEO: 12+3")) == "ROMEO: 12+3"


def test_eval_windows_are_fixed_and_held_out():
    w = data.eval_windows()
    assert w["old"][0].shape == w["new"][0].shape == (128, 128)
    assert torch.equal(w["old"][0][:, 1:], w["old"][1][:, :-1])         # targets are inputs shifted by one
    data.eval_windows.cache_clear()
    assert torch.equal(data.eval_windows()["new"][0], w["new"][0])
    held = data.texts()["a_val"]
    assert data.decode(w["old"][0][0]) in held


# ------------------------------------------------------------------ LoRA
def test_lora_starts_equal_and_counts():
    torch.manual_seed(0)
    model = forget.GPT(forget.config())
    x = torch.randint(0, len(data.vocab()), (2, 16))
    before = model(x)[0]
    new = lora.add_lora(model, 4)
    assert torch.allclose(model(x)[0], before)
    assert sum(p.numel() for p in new) == 24_576
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 24_576


def test_solution_b_lora():
    import solution_b_lora as sb

    torch.manual_seed(1)
    base = nn.Linear(16, 24)
    mine, ref = sb.LoRALinear(base, 4, 8.0), lora.LoRALinear(base, 4, 8.0)
    with torch.no_grad():
        mine.A.copy_(torch.randn(4, 16)); mine.B.copy_(torch.randn(24, 4))
        ref.A.copy_(mine.A); ref.B.copy_(mine.B)
    x = torch.randn(3, 16)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
    assert torch.allclose(mine.merged_weight(), ref.merged_weight(), atol=1e-6)
    for r in (4, 32):
        model = forget.GPT(forget.config())
        assert sb.lora_count(128, 3, r) == sum(p.numel() for p in lora.add_lora(model, r))


# ------------------------------------------------------------------ measures
def test_forward_kl_and_solution_a():
    import solution_a_kl as sa

    torch.manual_seed(2)
    p = F.log_softmax(torch.randn(4, 7, 91), -1)
    q = F.log_softmax(torch.randn(4, 7, 91), -1)
    assert forget.forward_kl(p, p) == pytest.approx(0.0, abs=1e-6)
    assert forget.forward_kl(p, q) > 0
    assert sa.forward_kl(p, q) == pytest.approx(forget.forward_kl(p, q), rel=1e-6)
    want = F.kl_div(q, p, log_target=True, reduction="none").sum(-1).mean()
    assert forget.forward_kl(p, q) == pytest.approx(float(want), rel=1e-5)
    assert sa.forgetting(2.0, 1.5) == pytest.approx(0.5)


def test_soft_loss_is_cross_entropy_plus_kl():
    torch.manual_seed(3)
    logits, ref = torch.randn(2, 5, 91), F.log_softmax(torch.randn(2, 5, 91), -1)
    y = torch.randint(0, 91, (2, 5))
    assert forget.soft_loss(logits, y, ref, 0.0) == pytest.approx(float(F.cross_entropy(logits.flatten(0, 1), y.flatten())))
    lp = F.log_softmax(logits, -1)
    kl = (ref.exp() * (ref - lp)).sum(-1).mean()
    ent = -(ref.exp() * ref).sum(-1).mean()
    assert forget.soft_loss(logits, y, ref, 1.0) == pytest.approx(float(kl + ent), rel=1e-5)


# ------------------------------------------------------------------ continual backprop
def test_solution_c_cbp_matches_src():
    import solution_c_cbp as sc

    torch.manual_seed(4)
    net = plasticity.Net()
    cbp = plasticity.CBP(net, rate=0.02, maturity=3)
    util = [u.clone() for u in cbp.util]
    for step in range(6):
        x = torch.rand(16, 784)
        _, hs = net(x, keep=True)
        hs = [h.detach() for h in hs]
        want = [sc.update_utility(util[i], hs[i], net.layers[i + 1].weight.detach(), cbp.decay) for i in range(2)]
        ages = [a.clone() + 1 for a in cbp.age]
        owed = [o + cbp.rate * len(u) for o, u in zip(cbp.owed, cbp.util)]
        picks = [sc.pick_units(want[i], ages[i], cbp.maturity, int(owed[i])) for i in range(2)]
        w_before = [lin.weight.detach().clone() for lin in net.layers]
        cbp.step(hs)
        for i in range(2):
            fresh = picks[i]
            keep = torch.ones(len(want[i]), dtype=torch.bool); keep[fresh] = False
            assert torch.allclose(cbp.util[i][keep], want[i][keep], atol=1e-6)
            assert torch.all(cbp.util[i][fresh] == 0)
            rows = torch.ones(net.layers[i + 1].out_features, dtype=torch.bool)
            if i == 0:
                rows[picks[1]] = False                 # those rows are new layer-1 units, re-drawn in full
            assert torch.all(net.layers[i + 1].weight[rows][:, fresh] == 0)
            if len(fresh):
                assert not torch.allclose(net.layers[i].weight[fresh], w_before[i][fresh])
        util = [u.clone() for u in cbp.util]
    assert cbp.replaced > 0


def test_health_counts_dead_units():
    net = plasticity.Net()
    with torch.no_grad():
        net.layers[0].bias[:64] = -1e6                 # 64 of 256 first-layer units can never fire
    h = plasticity.health(net, torch.rand(100, 784))
    assert h["dead"] >= 64 / 512
