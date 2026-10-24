"""The reference solutions in solutions/ agree with src/. No model; a few seconds."""

from __future__ import annotations

import math

import numpy as np
import pytest

import common
import database
import sandbox
import solution_a_pack as sa
import solution_b_sample as sb
import solution_c_dedupe as sc
import task


def test_pack_matches_task():
    for name in ("evolve", "shift"):
        for (sizes, cap), fn in [(inst, f) for inst in task.instances(name)[:2]
                                 for f in (task.first_fit, task.best_fit)]:
            assert sa.l1_bound(sizes, cap) == task.l1_bound(sizes, cap)
            assert sa.pack(sizes, cap, fn) == len(task.pack(sizes, cap, fn))


def test_pack_excess_of_best_fit():
    bins = [sa.pack(s, c, task.best_fit) for s, c in task.instances("evolve")]
    mean = np.mean([sa.excess(s, c, b) for (s, c), b in zip(task.instances("evolve"), bins)])
    assert mean == pytest.approx(task.excess(task.bins_used(task.best_fit, "evolve"), "evolve"))


@pytest.mark.parametrize("temperature", [0.25, 2.5, math.inf])
def test_sample_matches_database(temperature):
    db = database.Database(1, temperature, 10)
    excess = [4.781, 5.34, 4.159, 6.0]
    for i, x in enumerate(excess):
        db.add(database.Program(i, f"p{i}", x, (i,), 10), 0)
    _, probs = db.cluster_probs(0)
    assert np.allclose(probs, sb.cluster_probs(excess, temperature))


def test_sample_prediction():
    p = sb.cluster_probs([4.781, 5.340], 0.25)
    assert p[0] == pytest.approx(1 / (1 + math.exp(-(5.340 - 4.781) / 0.25)))


def test_dedupe_matches_sandbox():
    a = 'def priority(item, bins):\n    """Doc."""\n    # note\n    return -(bins - item)\n'
    b = "def priority(item, bins):\n    return -(bins-item)\n"
    assert sc.normalize(a) == sandbox.normalize(a) == sc.normalize(b)


def test_dedupe_counts_a_run():
    rows = common.read_jsonl(common.RUNS / "evolve_s0.jsonl")
    codes = [task.FIRST_FIT, task.BEST_FIT] + [r["code"] for r in rows if r["hash"]]
    assert sc.count_repeats(codes) == sum(r["status"] == "duplicate" for r in rows)
