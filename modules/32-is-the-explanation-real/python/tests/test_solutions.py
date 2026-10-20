"""The three reference solutions against the module's own code. CPU only, a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import scores  # noqa: E402
import solution_a_truth as sa  # noqa: E402
import solution_b_detection as sb  # noqa: E402
import solution_c_attribution as sc  # noqa: E402


def test_a_best_cosines_and_matching():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(16, 40)), rng.normal(size=(16, 40))
    assert np.allclose(sa.best_cosines(a, b), scores.best_cosines(a, b))
    assert np.allclose(sa.seed_match(a, b), scores.seed_match(a, b))
    perm = rng.permutation(40)
    assert np.allclose(sa.seed_match(a, a[:, perm] * 3.0), 1.0)          # scale and order do not matter


def test_a_on_the_model():
    from common import ARTIFACTS
    from model import load

    W = load(ARTIFACTS / "sparse.pt").blocks[1].mlp.W_out.weight.detach().numpy()
    rng = np.random.default_rng(1)
    assert sa.best_cosines(W, W[:, rng.permutation(W.shape[1])]).min() > 0.999


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b_rules_match(seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 30, size=6000)
    a = np.where(np.isin(t, [3, 4]) & (rng.random(6000) < 0.7), rng.random(6000), 0.0)
    s1, e1 = sb.detection(a, t)
    s2, e2 = scores.detection(a, t)
    assert s1 == pytest.approx(s2) and list(e1) == list(e2)
    assert sb.token_r2(a, t, 30) == pytest.approx(scores.token_r2(a, t, 30))


def test_b_on_true_neurons():
    import data
    from analyse_dicts import to_csc, true_codes
    from dicts import get_model

    _, va = data.tokens()
    m = get_model("trained", "cpu")
    ids = va[: 2**18]
    T = to_csc(*true_codes(m, va, "cpu"), m.c.n_mlp)
    checked = 0
    for i in range(0, 2048, 37):
        a = T[:, i].toarray().ravel()
        s1, _ = sb.detection(a, ids)
        s2, _ = scores.detection(a, ids)
        assert (s1 is None) == (s2 is None)
        if s1 is not None:
            assert s1 == pytest.approx(s2)
            assert sb.token_r2(a, ids, 4096) == pytest.approx(scores.token_r2(a, ids, 4096))
            checked += 1
    assert checked >= 20


@pytest.mark.parametrize("which", ["trained", "random"])
def test_c_attribution_and_ablation(which):
    import attribution
    import data
    from dicts import get_model

    _, va = data.tokens()
    sep = data.Vocab().encode("<|endoftext|>")[0]
    x = torch.from_numpy(attribution.prompts(va, sep)[0])[None]
    m = get_model(which, "cpu")
    with torch.no_grad():
        target = int(m(x)[0, -1].argmax())
    att, logit = sc.attribute(m, x, target)
    _, _, ref, ref_logit = attribution.attribute(m, x, target)
    assert np.allclose(att, ref, atol=1e-5) and logit == pytest.approx(ref_logit, abs=1e-4)
    l, p, n = np.unravel_index(np.argmax(np.abs(att)), att.shape)
    drop = sc.ablate(m, x, target, int(l), int(p), int(n))
    ref_drop = logit - attribution.run_masked(m, x, [[(int(l), int(p), int(n))]], target)[0]
    assert drop == pytest.approx(float(ref_drop), abs=1e-4)
