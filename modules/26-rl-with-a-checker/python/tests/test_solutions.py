"""The task, the checkers and the reference solutions, against src/. CPU only, a few seconds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402,F401  (laptop guard before torch)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import library  # noqa: E402
import rl  # noqa: E402
import task  # noqa: E402


# ------------------------------------------------------------------ the task
def test_formats_and_checker():
    assert task.prompt(374, 485) == "374+485="
    assert task.direct(374, 485) == "859."
    assert task.worked(374, 485) == "[4+5=9,7+8+0=15,3+4+1=8]859."
    assert len(task.worked(999, 999)) == 31 <= task.MAX_REPLY
    for a, b in [(374, 485), (999, 999), (100, 100), (505, 495)]:
        assert task.check(task.direct(a, b), a, b) == 1
        assert task.check(task.worked(a, b), a, b) == 1
    assert task.check("859", 374, 485) == 0                       # never ended
    assert task.check("858.", 374, 485) == 0
    assert task.check("[1+1=2]859.", 374, 485) == 1                # wrong steps, right number: the checker reads the end only
    assert task.check("[4+5=9]8590.", 374, 485) == 0
    assert task.reply_text(task.encode("859.__")) == "859."
    assert task.is_worked("[4+5") and not task.is_worked("859.")


def test_splits_hold_out_the_test_pairs():
    test, draw = task.splits()
    held = {tuple(p) for p in test.tolist()}
    assert len(test) == 500 and test.min() >= 100 and test.max() <= 999
    drawn = draw(5000, np.random.default_rng(3))
    assert not any(tuple(p) in held for p in drawn.tolist())


def test_library_checker():
    assert library.check("859", 374, 485) == 1
    assert library.check("The answer is 859.", 374, 485) == 1
    assert library.check("374 + 485 = 859", 374, 485) == 1          # the last number counts
    assert library.check("859, or maybe 860", 374, 485) == 0
    assert library.check("The sum is 859,", 374, 485) == 1
    assert library.check("1,013", 514, 499) == 1
    assert library.check("85 9", 374, 485) == 0
    assert library.check("no idea", 374, 485) == 0
    assert library.reward_fn([[{"content": "859"}], "860"], a=[374, 374], b=[485, 485]) == [1.0, 0.0]


# ---------------------------------------------------------- (a) advantages
def test_advantages_match_src():
    import solution_a_advantages as sol

    r = (torch.rand(50, 8, generator=torch.Generator().manual_seed(0)) < 0.6).float()
    assert torch.allclose(sol.adv_group(r), rl.adv_group(r))
    assert torch.allclose(sol.adv_no_std(r), rl.adv_no_std(r))
    assert torch.allclose(sol.adv_loo(r), rl.adv_loo(r))
    # RLOO is Dr. GRPO times G / (G - 1)
    assert torch.allclose(rl.adv_loo(r), rl.adv_no_std(r) * 8 / 7, atol=1e-6)
    # an all-right or all-wrong group teaches nothing
    assert rl.adv_group(torch.ones(1, 8)).abs().max() == 0


# ----------------------------------------------------------- (b) pass@k
def test_pass_at_k_matches_src_and_results():
    import solution_b_pass_at_k as sol

    for n, c, k in [(128, 0, 1), (128, 5, 1), (128, 5, 64), (128, 127, 2), (16, 3, 16), (16, 0, 16)]:
        assert sol.pass_at_k(n, c, k) == pytest.approx(task.pass_at_k(n, c, k))
    assert task.pass_at_k(128, 64, 1) == pytest.approx(0.5)
    base = json.loads((ROOT / "results" / "runs" / "base_eval.json").read_text())["final"]
    ks = [int(k) for k in base["pass_at_k"]]
    got = sol.curve(base["right_per_problem"], base["n_samples"], ks)
    assert got == pytest.approx([base["pass_at_k"][str(k)] for k in ks])


# ----------------------------------------------------------------- (c) loss
def test_loss_matches_src():
    import solution_c_loss as sol

    g = torch.Generator().manual_seed(1)
    N, T = 16, task.MAX_REPLY
    mask = (torch.arange(T)[None, :] < torch.randint(2, T, (N, 1), generator=g)).float()
    lp_old = -torch.rand(N, T, generator=g)
    lp = lp_old + 0.3 * torch.randn(N, T, generator=g)            # ratios away from 1: the clip acts
    adv = torch.randn(N, generator=g)
    for norm in (True, False):
        assert sol.pg_loss(lp, lp_old, adv, mask, norm) == pytest.approx(float(rl.pg_loss(lp, lp_old, adv, mask, norm)))


def test_length_weighting():
    """Under GRPO a character of a 28-character reply gets 1/7 the push of one in a 4-character reply."""
    T = task.MAX_REPLY
    mask = torch.zeros(2, T)
    mask[0, :4] = 1
    mask[1, :28] = 1
    adv = torch.tensor([-1.0, -1.0])
    for norm, ratio in ((True, 7.0), (False, 1.0)):
        lp = torch.zeros(2, T, requires_grad=True)
        rl.pg_loss(lp, lp.detach(), adv, mask, norm).backward()
        assert float(lp.grad[0, 0] / lp.grad[1, 0]) == pytest.approx(ratio)
