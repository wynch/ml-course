"""The reference solutions, checked against the scripts and the committed results.

(a) and (b) run on the CPU in a second. (c) needs the cached pairs
(src/data.py fetches them, no model) and is skipped without them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# ------------------------------------------------------------------ (a) bt
def test_bt_grad_matches_script_and_finite_difference():
    import bt_synthetic as bt
    import solution_a_bt as sol

    rng = np.random.default_rng(0)
    s = rng.normal(size=12)
    w, l = bt.simulate(bt.true_scores(), 200, seed=3)
    assert np.allclose(sol.bt_grad(s, w, l), bt.bt_grad(s, w, l))
    e = np.zeros(12); e[4] = 1e-6
    fd = (bt.bt_loss(s + e, w, l) - bt.bt_loss(s - e, w, l)) / 2e-6
    assert abs(fd - bt.bt_grad(s, w, l)[4]) < 1e-6


def test_bt_fit_recovers_ranking():
    import bt_synthetic as bt
    import solution_a_bt as sol

    w, l = bt.simulate(bt.true_scores(), 1000, seed=0)
    s = sol.fit(w, l, bt.N_ITEMS)
    assert np.allclose(s, common.read_results("bt_synthetic")["main"]["s_fit"], atol=1e-3)
    assert bt.kendall_tau(bt.true_scores(), s) > 0.9


# ------------------------------------------------------------- (b) dpo loss
def test_reply_logp_matches_cross_entropy():
    import solution_b_dpo_loss as sol

    torch.manual_seed(0)
    logits, ids = torch.randn(20, 50), torch.randint(0, 50, (20,))
    ref = -F.cross_entropy(logits[:-1], ids[1:], reduction="none")[6:].sum()
    assert torch.allclose(sol.reply_logp(logits, ids, 7), ref, atol=1e-5)


def test_dpo_loss_matches_stored_batch_and_trl():
    import solution_b_dpo_loss as sol

    r = common.read_results("dpo_by_hand")
    t = lambda k: torch.tensor([p[k] for p in r["per_pair"]], dtype=torch.float64)  # noqa: E731
    loss = float(sol.dpo_loss(t("pi_chosen"), t("pi_rejected"), t("ref_chosen"), t("ref_rejected"), r["beta"]))
    assert loss == pytest.approx(r["hand"]["loss"], abs=1e-9)
    assert loss == pytest.approx(r["trl"]["loss"], abs=1e-3)
    same = float(sol.dpo_loss(t("ref_chosen"), t("ref_rejected"), t("ref_chosen"), t("ref_rejected")))
    assert same == pytest.approx(math.log(2))


# ------------------------------------------------------- (c) length reward
def _cached() -> bool:
    import data
    return (data.cache_dir() / f"filtered-{data.MAX_TOKENS}.json").exists()


def test_length_reward():
    if not _cached():
        pytest.skip("pairs not cached; run src/data.py first")
    import data
    import solution_c_length_reward as sol

    sp = data.splits(data.load_pairs()[0])
    d = sol.length_feature(sp["reward_train"])
    w = sol.fit_w(d)
    grad = np.mean(-(1 - 1 / (1 + np.exp(-w * d))) * d)
    assert abs(grad) < 1e-6                                        # a minimum
    d_te = sol.length_feature(sp["heldout"])
    acc = float(np.mean(w * d_te > 0) + 0.5 * np.mean(d_te == 0))
    assert acc == pytest.approx(common.read_results("length_bias")["length_only_reward"]["heldout_accuracy"], abs=1e-4)
