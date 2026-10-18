"""The reference solutions, checked against src/ and the committed results. No model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402
import gsm8k  # noqa: E402
from test_extract import CASES  # noqa: E402


@pytest.fixture(scope="module")
def analysis():
    return common.read_results("analysis")


def test_extract_known_answers():
    import solution_a_extract as sol

    for text, want in CASES:
        got = sol.extract(text)
        assert (got is None and want is None) or gsm8k.same(got, want), (text, got, want)


def test_extract_matches_src_on_every_cached_reply():
    import solution_a_extract as sol

    for name, key in (("greedy", None), ("samples", "samples")):
        for r in common.read_jsonl(common.GEN / f"{name}.jsonl"):
            texts = [s["text"] for s in r[key]] if key else [r["text"]]
            for t in texts:
                assert sol.extract(t) == gsm8k.extract(t)


def test_vote_matches(analysis):
    import solution_b_vote as sol

    assert sol.vote(["3", None, "4", "4", "3"]) == "3"        # tie: the first seen wins
    assert sol.vote([None, None]) is None
    for n in (1, 3, 8):
        assert abs(sol.vote_accuracy(analysis["per_problem"], n)
                   - analysis["section2"]["vote_mean_over_subsets"][str(n)]) < 1e-12


def test_best_of_n_matches(analysis):
    import solution_c_best_of_n as sol

    assert sol.best_of(["1", "2", "3"], [0.2, 0.9, 0.9]) == "2"
    assert sol.weighted_vote(["1", "2", "2"], [0.9, 0.4, 0.4]) == "1"
    for n in (2, 8):
        assert abs(sol.mean_accuracy(analysis["per_problem"], n, sol.best_of, "logodds")
                   - analysis["section3"]["best_mean_over_subsets"][str(n)]) < 1e-12
        assert abs(sol.mean_accuracy(analysis["per_problem"], n, sol.weighted_vote, "scores")
                   - analysis["section3"]["weighted_vote_mean_over_subsets"][str(n)]) < 1e-12
