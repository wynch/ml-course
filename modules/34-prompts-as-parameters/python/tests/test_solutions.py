"""The reference solutions, checked against src/ and the committed results. No model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import analyse  # noqa: E402
import common  # noqa: E402
import optimise  # noqa: E402


def test_pareto_matches_src_on_random_pools():
    import solution_a_pareto as sol

    rng = np.random.default_rng(0)
    for _ in range(200):
        n, q = rng.integers(1, 8), rng.integers(1, 12)
        rows = rng.integers(0, 2, size=(n, q)).tolist()
        pool = [{"id": str(i), "right": r, "acc": float(np.mean(r))} for i, r in enumerate(rows)]
        want = {int(k): v for k, v in optimise.pareto_parents(pool).items()}
        assert sol.pareto_parents(rows) == want


def test_pareto_example():
    import solution_a_pareto as sol

    rows = [[1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    assert sol.pareto_parents(rows) == {0: 4, 1: 2}


def test_sign_test_matches_analysis():
    import solution_b_sign_test as sol

    base = common.read_results("baseline")
    a = common.read_results("analysis")
    got = sol.sign_test(base["baseline"]["test_right"], base["by_hand"]["test_right"])
    assert got == a["by_hand"]["test_vs_baseline"]
    for r in a["runs"]:
        run = common.read_results(f"runs/{r['method']}_s{r['seed']}")
        assert sol.sign_test(base["baseline"]["test_right"], run["best"]["test_right"]) == r["test_vs_baseline"]


def test_sign_test_edges():
    import solution_b_sign_test as sol

    assert sol.sign_test([1, 0], [1, 0]) == {"wins": 0, "losses": 0, "p": 1.0}
    assert sol.sign_test([0] * 10, [1] * 10)["p"] == 2 / 2 ** 10
    assert sol.sign_test([0, 1], [1, 0]) == analyse.sign_test([0, 1], [1, 0])


def test_curve_matches_run_files():
    import solution_c_curve as sol

    for m in optimise.METHODS:
        for s in optimise.SEEDS:
            rows = common.read_jsonl(common.RUNS / f"{m}_s{s}.jsonl")
            run = common.read_results(f"runs/{m}_s{s}")
            curve = sol.best_so_far(rows)
            assert [[c, a] for c, a in curve] == [[c[0], c[1]] for c in run["curve"]]
            assert curve[-1][1] == run["best"]["train"]
            assert sol.budget_to_best(rows) <= optimise.BUDGET
