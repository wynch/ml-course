"""The committed logs and results/analysis.json: complete, inside the laptop budget, consistent.

No model is loaded. A few kept programs are re-run through the sandbox, which
takes a second or two.
"""

from __future__ import annotations

import json
import re

import pytest

import analyse
import common
import evolve
import sandbox
import task

A = common.read_results("analysis")
RUN_NAMES = sorted(A["runs"])
NOVERIFIER = [n for n in RUN_NAMES if n.startswith("noverifier")]


def rows(name):
    return common.read_jsonl(common.RUNS / f"{name}.jsonl")


def test_core_runs_present_at_full_budget():
    for name in ("evolve_s0", "evolve_s1", "evolve_s2", "random_s0", "random_s1", "random_s2",
                 "noselect_s0", "noverifier_s0", "evolve_flat_s0"):
        assert name in A["runs"], name
    for name in RUN_NAMES:
        # The no-verifier runs were stopped early (README, section 4).
        want = evolve.BUDGET if name not in NOVERIFIER else len(rows(name))
        assert len(rows(name)) == want == A["runs"][name]["candidates"]
        assert want % evolve.SAMPLES_PER_PROMPT == 0 and want >= 200


@pytest.mark.parametrize("name", RUN_NAMES)
def test_guard_budget(name):
    guards = common.read_jsonl(common.RUNS / f"{name}.guards.jsonl")
    assert guards, name
    for g in guards:
        assert g["minutes"] < 5, name                       # every chunk under five minutes
        assert g["peak_rss_gb"] + g["peak_metal_gb"] < 6, name


@pytest.mark.parametrize("name", RUN_NAMES)
def test_status_counts_and_curve_recompute(name):
    r = rows(name)
    run = A["runs"][name]
    counts = {}
    for row in r:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    assert counts == run["status"]
    assert run["best_so_far"] == analyse.best_so_far(r, A["baselines"]["best_fit"]["evolve"])
    for row in r:
        if row["status"] == "valid":
            assert row["excess"] == pytest.approx(task.excess(row["bins"], "evolve"))


def test_noverifier_never_reads_scores():
    for name in NOVERIFIER:
        r = rows(name)
        assert all(not row["seen_by_loop"] for row in r)
        assert all(row["status"] != "duplicate" for row in r)    # no dedupe either


def test_duplicates_are_exact_repeats():
    for name in RUN_NAMES:
        if name.startswith("noverifier"):
            continue
        r = rows(name)
        seen = {sandbox.code_hash(c) for c in (task.FIRST_FIT, task.BEST_FIT)} if not name.startswith("random") else set()
        for row in r:
            if row["status"] == "duplicate":
                assert row["hash"] in seen, (name, row["idx"])
            elif row["hash"]:
                seen.add(row["hash"])


def test_baselines():
    for name, fn in (("first_fit", task.first_fit), ("best_fit", task.best_fit)):
        for s in task.SETS:
            assert A["baselines"][name][s] == pytest.approx(task.excess(task.bins_used(fn, s), s), abs=1e-4)


@pytest.mark.skipif(not sandbox.OS_SANDBOX, reason="candidates run only under macOS sandbox-exec")
@pytest.mark.parametrize("name", ["evolve_s0", "random_s1", "noselect_s0"])
def test_kept_programs_rescore(name):
    kept = A["runs"][name]["kept"]
    out = sandbox.evaluate(kept["code"], sets=("evolve", "heldout", "shift"))
    assert out["status"] == "valid"
    for s, b in out["bins"].items():
        assert task.excess(b, s) == pytest.approx(kept["excess"][s], abs=1e-4)


def test_summary_means():
    for arm, entry in A["summary"].items():
        kept = [A["runs"][n]["kept"]["excess"] for n in entry["runs"] if A["runs"][n]["kept"]["excess"]]
        for s in task.SETS:
            if kept:
                assert entry[s]["mean"] == pytest.approx(sum(k[s] for k in kept) / len(kept), abs=1e-4)


def test_expectations_scored():
    exp = common.read_results("expectations")
    scored = common.read_results("scored")
    assert set(scored["verdicts"]) == set(exp["expectations"])
    for v in scored["verdicts"].values():
        assert v["verdict"] in ("met", "missed")


def test_explorable_matches_results():
    page = (common.MODULE.parents[1] / "explorables" / "35-evolution-with-an-llm-mutator.html").read_text()
    blob = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1)
    data = json.loads(blob)
    assert data["budget"] == A["budget"]
    assert data["curves"]["evolve_s0"] == A["runs"]["evolve_s0"]["best_so_far"]
    assert len(data["stepper"]) == evolve.BUDGET // evolve.SAMPLES_PER_PROMPT
    assert data["stepper"][-1]["best"] == pytest.approx(min(A["runs"]["evolve_s0"]["best_so_far"]), abs=1e-3)
    assert f'data-value="{data["predict_truth"]}" id="predict-answer"' in page
    for meta in ('name="lk-app" content="ml-course"', 'name="lk-build" content="dev"', 'name="lk-sw" content="../sw.js"'):
        assert meta in page
    assert 'href="../learnkit/lk.css"' in page and '"../learnkit/lk.js"' in page
    for tag in ("<lk-deck", "<lk-predict", "<lk-stepper"):
        assert tag in page


def test_quiz_json():
    path = common.MODULE.parents[1] / "quizzes" / "35.json"
    quiz = json.loads(path.read_text())
    assert quiz["id"] == "ml-course-35"
    qs = quiz["questions"]
    assert len(qs) == 10 and len({q["id"] for q in qs}) == 10
    positions = []
    for q in qs:
        assert q["kind"] in ("single", "multi", "numeric") and q["explain"]
        if q["kind"] == "single":
            assert 0 <= q["answer"] < len(q["options"])
            positions.append(q["answer"])
        elif q["kind"] == "multi":
            assert all(0 <= a < len(q["options"]) for a in q["answer"])
    assert len(set(positions)) >= 3                      # right answers spread over positions
    page = (path.parent / "35.html").read_text()
    assert 'src="35.json"' in page and 'QUIZ_ID = "35"' in page
