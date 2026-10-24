"""The committed results: complete, inside the budgets, consistent, and in the explorable.

No model is loaded; these run in about a second.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import analyse  # noqa: E402
import common  # noqa: E402
import optimise  # noqa: E402

RUN_NAMES = [f"{m}_s{s}" for m in optimise.METHODS for s in optimise.SEEDS]
EXPLORABLE = ROOT.parents[1] / "explorables" / "34-prompts-as-parameters.html"
QUIZ = ROOT.parents[1] / "quizzes" / "34.json"


def guards():
    yield "baseline", common.read_results("baseline")["guard"]
    for name in RUN_NAMES:
        for i, g in enumerate(common.read_results(f"runs/{name}")["chunks"]):
            yield f"{name}_{i}", g
    for name in ("library_evaluate", "library_score"):
        if (common.RESULTS / f"{name}.json").exists():
            yield name, common.read_results(name)["guard"]


@pytest.mark.parametrize("name,g", list(guards()))
def test_guard_budget(name, g):
    assert g["minutes"] < 5, name                        # every chunk under five minutes
    assert g["peak_rss_gb"] < 6 and g["peak_metal_gb"] < 8, name   # OPRO peaked at 7.07 GB


@pytest.mark.parametrize("name", RUN_NAMES)
def test_run_complete_and_within_budget(name):
    run = common.read_results(f"runs/{name}")
    assert run["calls"] <= optimise.BUDGET
    assert optimise.BUDGET - run["calls"] < 60 + optimise.GEPA_MINI   # the budget was used up
    rows = common.read_jsonl(common.RUNS / f"{name}.jsonl")
    full = [r for r in rows if r["event"] == "full"]
    assert full[0]["text"] == "Classify the question by the type of its answer."
    assert max(r["acc"] for r in full) == run["best"]["train"]
    best = next(r for r in full if r["acc"] == run["best"]["train"])
    assert best["text"] == run["best"]["text"]           # the earliest of the best
    assert len(run["best"]["test_right"]) == 500 and len(run["best"]["heldout_right"]) == 300
    assert abs(sum(run["best"]["test_right"]) / 500 - run["best"]["test"]) < 1e-9


def test_analysis_recomputes(tmp_path, monkeypatch):
    before = common.read_results("analysis")
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    (tmp_path / "runs").symlink_to(ROOT / "results" / "runs")
    for f in (ROOT / "results").glob("*.json"):
        if f.name != "analysis.json":
            (tmp_path / f.name).write_text(f.read_text())
    monkeypatch.setattr(common, "RUNS", tmp_path / "runs")
    analyse.main()
    after = json.loads((tmp_path / "analysis.json").read_text())
    assert after == before


def test_splits_disjoint_sizes():
    b = common.read_results("baseline")
    assert b["sizes"] == {"train": 60, "heldout": 300, "test": 500}
    assert b["label_counts"]["train"] == [10] * 6 and b["label_counts"]["heldout"] == [50] * 6


def test_explorable_data_matches():
    html = EXPLORABLE.read_text()
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    a = common.read_results("analysis")
    assert all(data["baseline"][k] == a["baseline"][k] for k in ("train", "heldout", "test"))
    for r in data["runs"]:
        want = next(x for x in a["runs"] if x["method"] == r["method"] and x["seed"] == r["seed"])
        assert r["test"] == want["test"] and r["text"] == want["text"]
    assert "../learnkit/lk.js" in html and 'name="lk-build" content="dev"' in html


def test_quiz_shape():
    q = json.loads(QUIZ.read_text())
    assert q["id"] == "ml-course-34" and 8 <= len(q["questions"]) <= 10
    ids = [x["id"] for x in q["questions"]]
    assert len(set(ids)) == len(ids)
    for x in q["questions"]:
        assert x["kind"] in ("single", "multi", "numeric") and x["explain"]
        if x["kind"] == "single":
            assert 0 <= x["answer"] < len(x["options"])
