"""The reference solutions, checked against the scripts and the committed results.

Tests (a) use module 04's tiny GPT on the CPU and take a second. Tests (b) and
(c) load SmolLM2-360M once (it is cached in ~/.cache/huggingface after
module 05) and run a handful of short forward passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)
import torch  # noqa: E402


# --------------------------------------------------------------- (a) hooks
@pytest.fixture(scope="module")
def tiny():
    from tiny_gpt import load_tiny_gpt

    return load_tiny_gpt("cpu")


def test_hooks_read_matches_script(tiny):
    import hooks
    import solution_a_hooks as sol

    model, tok = tiny
    ids = torch.tensor(tok.encode(sol.TEXT))
    mine = sol.record_neuron(model, ids, sol.BLOCK, sol.NEURON)
    ref = hooks.record_neurons(model, ids)[sol.BLOCK][:, sol.NEURON]
    assert torch.allclose(mine, ref, atol=1e-6)
    spaces = torch.tensor([c == " " for c in sol.TEXT])
    assert mine[spaces].min() > mine[~spaces].max()      # it is the space neuron


def test_hooks_are_removed_and_edit_works(tiny):
    import hooks
    import solution_a_hooks as sol

    model, tok = tiny
    ids = torch.tensor(tok.encode(sol.TEXT))
    with torch.no_grad():
        plain = model(ids[None])[0][0]
    same = sol.scale_neuron_logits(model, ids, sol.BLOCK, sol.NEURON, 1.0)
    assert torch.allclose(same, plain, atol=1e-6)        # k = 1 changes nothing
    zeroed = sol.scale_neuron_logits(model, ids, sol.BLOCK, sol.NEURON, 0.0)
    handles = hooks.ablate(model, {sol.BLOCK: [sol.NEURON]})
    try:
        with torch.no_grad():
            ref = model(ids[None])[0][0]
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(zeroed, ref, atol=1e-5)        # post-GELU x0 == pre-GELU := 0
    with torch.no_grad():
        again = model(ids[None])[0][0]
    assert torch.allclose(again, plain)                  # no hook left behind


# ------------------------------------------------- (b) and (c): SmolLM2-360M
@pytest.fixture(scope="module")
def smol():
    from common import load_hf_model, load_tokenizer

    return load_hf_model(), load_tokenizer()


def test_probe_plain_hooks_match_nnsight(smol):
    import probe
    import solution_b_plain_probe as sol

    model, tok = smol
    plain = sol.last_token_residuals_plain(model, tok, sol.SENTENCES)
    traced = probe.last_token_residuals(model, tok, sol.SENTENCES, batch=4)
    assert plain.shape == traced.shape == (4, 33, 960)
    scale = np.abs(plain).max(axis=-1, keepdims=True)
    assert (np.abs(plain - traced) / scale).max() < 1e-3  # left padding changes nothing


def test_probe_plateau_rule():
    import probe
    import solution_b_plain_probe as sol

    acc = [0.5, 0.6, 0.86, 0.80, 0.87, 0.88, 0.86]
    assert sol.plateau_layer(acc) == 4
    assert sol.plateau_layer(acc, 3.0) == 2
    res = common.read_results("probe")
    assert sol.plateau_layer(res["accuracy_trained"]) == probe.plateau_layer(res["accuracy_trained"])
    assert sol.plateau_layer(res["accuracy_trained"]) == res["plateau_layer"]


def test_patch_cells(smol):
    import solution_c_patch_cell as sol

    model, tok = smol
    assert sol.normalised_score(3.0, 3.0, -1.0) == 1.0
    assert sol.normalised_score(-1.0, 3.0, -1.0) == 0.0
    top = sol.patch_cell(model, tok, 32, 9)              # whole final residual at the last token
    assert abs(top["score"] - 1.0) < 1e-3
    same_token = sol.patch_cell(model, tok, 0, 0)        # 'The' is the same in both prompts
    assert abs(same_token["score"]) < 1e-4
    sweep = common.read_results("patching")["score"]
    for row, pos in ((3, 2), (27, 9)):
        assert abs(sol.patch_cell(model, tok, row, pos)["score"] - sweep[row][pos]) < 0.02
