"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

No model is trained or loaded here; these run in a second or two.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402
import task  # noqa: E402

RUNS = common.RESULTS / "runs"
ARMS = ["grpo", "drgrpo", "rloo", "random", "random_clip"]
SEEDS = [0, 1, 2]


def run(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def guards():
    yield "sft", common.read_results("sft")["guard"]
    yield "base_eval", run("base_eval")["guard"]
    for a in ARMS:
        for s in SEEDS:
            yield f"{a}_s{s}", run(f"{a}_s{s}")["guard"]
    for name in ("library_base", "library_base_first_number", "library_after"):
        if (common.RESULTS / f"{name}.json").exists():
            yield name, common.read_results(name)["guard"]
    if (common.RESULTS / "library_train.json").exists():
        for i, c in enumerate(common.read_results("library_train")["chunks"]):
            yield f"library_train_{i}", c["guard"]


@pytest.mark.parametrize("name,g", list(guards()))
def test_guard_budget(name, g):
    assert g["minutes"] < 5, name                      # every script or chunk under five minutes
    assert g["peak_rss_gb"] + g["peak_mps_driver_gb"] < 4, name
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_all_runs_present():
    for a in ARMS:
        for s in SEEDS:
            r = run(f"{a}_s{s}")
            assert r["arm"] == a and r["seed"] == s and r["config"]["steps"] == 300
            assert len(r["final"]["right_per_problem"]) == task.N_TEST


def test_pass_at_k_recomputes():
    for name in ["base_eval"] + [f"{a}_s{s}" for a in ARMS for s in SEEDS]:
        f = run(name)["final"]
        for k, v in f["pass_at_k"].items():
            want = sum(task.pass_at_k(f["n_samples"], c, int(k)) for c in f["right_per_problem"]) / task.N_TEST
            assert v == pytest.approx(want), (name, k)
        assert f["pass_at_k"]["1"] == pytest.approx(sum(f["right_per_problem"]) / (task.N_TEST * f["n_samples"]))


def test_summary_matches_runs():
    R = common.read_results("rl")
    for a in ARMS:
        seeds = R["arms"][a]["pass_at_k"]["1"]["seeds"]
        assert seeds == pytest.approx([run(f"{a}_s{s}")["final"]["pass_at_k"]["1"] for s in SEEDS])
    assert R["base"]["pass_at_k"] == run("base_eval")["final"]["pass_at_k"]
    assert set(R["expectations_met"]) == {f"E{i}" for i in range(1, 9)}


def test_library_summary():
    R = common.read_results("rl")["library"]
    b, a = common.read_results("library_base")["measure"], common.read_results("library_after")["measure"]
    assert R["pass_at_k_after"] == a["pass_at_k"] and R["pass_at_k_before"] == b["pass_at_k"]
    for m in (a, b):
        assert m["pass_at_k"]["1"] == pytest.approx(sum(m["right_per_problem"]) / (m["problems"] * m["samples"]))
    assert common.read_results("library_train")["chunks"][-1]["reached_step"] == R["steps"]


def test_sft_base():
    S = common.read_results("sft")
    assert S["config"]["steps"] == 3000 and S["config"]["worked_share"] == 0.2
    assert (common.ARTIFACTS / "base.pt").stat().st_size < 300_000
    assert 0.3 < S["test"]["sample_acc"] < 0.8          # the base is right often, not always


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
