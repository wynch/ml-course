"""The reference solutions, checked against the scripts.

(a) and (b) run on the CPU in a second or two. (c) needs the GPT-2 SAE file
in the Hugging Face cache (src/gpt2_sae.py fetches it) and is skipped
without it.
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


# ----------------------------------------------------------------- (a) toy
def test_toy_forward_matches_script():
    import solution_a_toy as sol
    import toy

    g = torch.Generator().manual_seed(1)
    W, b, x = torch.randn(2, 5, generator=g), torch.randn(5, generator=g), torch.rand(64, 5, generator=g)
    imp = torch.tensor([0.9**i for i in range(5)])
    ref = toy.forward(W[None], b[None], x[None])[0]
    assert torch.allclose(sol.toy_forward(W, b, x), ref, atol=1e-6)
    assert torch.allclose(sol.toy_loss(W, b, x, imp), toy.loss_fn(W[None], b[None], x[None], imp)[0], atol=1e-6)


def test_toy_represented_on_results():
    import solution_a_toy as sol

    for setup in common.read_results("toy")["setups"].values():
        for f in setup["frames"]:
            assert sol.represented(torch.tensor(f["W"])) == f["n_represented"]


def test_toy_trains_dense_vs_sparse():
    import solution_a_toy as sol

    assert sol.represented(sol.train_one(0.0)) == 2
    assert sol.represented(sol.train_one(0.99)) == 5


# --------------------------------------------------------------- (b) topk
def test_topk_matches_script():
    import sae
    import solution_b_topk as sol

    torch.manual_seed(0)
    x = torch.randn(32, 512)
    model = sae.TopKSAE(512, 2048, 32, torch.randn(512) * 0.1)
    with torch.no_grad():
        model.b_enc.normal_(0, 0.1)
        ref = model.encode(x)
    mine = sol.topk_encode(x, model.W_enc.detach(), model.b_enc.detach(), model.b_pre.detach(), 32)
    assert torch.allclose(mine, ref, atol=1e-6)
    assert int((mine > 0).sum(-1).max()) <= 32


def test_fve_and_projection():
    import solution_b_topk as sol

    x = torch.randn(100, 8)
    assert sol.fve(x, x) == pytest.approx(1.0)
    assert sol.fve(x, x.mean(0).expand_as(x)) == pytest.approx(0.0, abs=1e-6)
    W = torch.nn.functional.normalize(torch.randn(8, 20), dim=0)
    g = sol.project_grad(torch.randn(8, 20), W)
    assert torch.allclose((g * W).sum(0), torch.zeros(20), atol=1e-5)   # no part along the column


# ------------------------------------------------------------ (c) by hand
def _cached() -> bool:
    from huggingface_hub import try_to_load_from_cache
    import solution_c_sae_by_hand as sol

    return isinstance(try_to_load_from_cache(sol.REPO, sol.FILE), str)


def test_sae_by_hand_matches_saelens():
    import solution_c_sae_by_hand as sol

    if not _cached():
        pytest.skip("GPT-2 SAE not in the Hugging Face cache; run src/gpt2_sae.py first")
    from sae_lens import SAE

    mine = sol.load_sae()
    lib = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cpu")
    torch.manual_seed(0)
    x = mine["b_dec"] + 3 * torch.randn(4, 768)
    with torch.no_grad():
        f_lib = lib.encode(x)
        assert torch.allclose(sol.encode(mine, x), f_lib, atol=1e-3)
        assert torch.allclose(sol.decode(mine, f_lib), lib.decode(f_lib), atol=1e-3)
    one = sol.encode(mine, mine["b_dec"] + 40 * mine["W_dec"][sol.LATENT])
    assert int(one.argmax()) == sol.LATENT
