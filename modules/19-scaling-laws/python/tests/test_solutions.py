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
