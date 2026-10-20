"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

No model is trained here; these run in a few seconds.
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

GB_CAP = 4.0          # the module's peak-memory guard


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_every_chunk_inside_the_guard():
    for folder in ("runs", "plan_runs", "library_runs"):
        for g in _jsonl(common.RESULTS / folder / "guards.jsonl"):
            assert g["minutes"] < 5, (folder, g["script"])
            assert g["peak_rss_gb"] + g["peak_mps_driver_gb"] < GB_CAP, (folder, g["script"])
            assert g["mps_high_watermark_ratio"] == "0.5"
    for name in ("world", "train", "plan", "grid"):
        g = common.read_results(name)["guard"]
        assert g["minutes"] < 5 and g["peak_rss_gb"] < GB_CAP
    for arm in common.read_results("grid")["arms"].values():
        assert arm["minutes"] < 5 and arm["peak_rss_gb"] < GB_CAP


def test_expectations_written_and_complete():
    E = json.loads((common.RESULTS / "expectations.json").read_text())
    assert len(E["expectations"]) == 20 and E["written"].startswith("2026-10-20, before")


def test_train_runs_complete():
    T = common.read_results("train")
    assert set(T["arms"]) == {"lewm", "pred_only", "dim8", "dim192"}
    for arm, r in T["arms"].items():
        assert r["seeds"] == [0, 1, 2], arm
        assert len(r["probe_r2_xy"]) == 3 and all(-1 <= v <= 1 for v in r["probe_r2_xy"])
    for f in (common.RESULTS / "runs").glob("*_s*.json"):
        r = json.loads(f.read_text())
        assert r["progress"]["step"] == 3000, f.name
    assert (common.ARTIFACTS / "lewm_planner.pt").stat().st_size < 1_000_000


def test_plan_rows_complete_and_consistent():
    P = common.read_results("plan")
    R = P["rows"]
    assert len(R) == 22
    for name, r in R.items():
        for s in ("cross", "same"):
            assert len(r[s]["per_task"]) == 100, name
            assert abs(sum(r[s]["per_task"]) / 100 - r[s]["success"]) < 1e-9
    assert R["expert__none"]["cross"]["success"] == 1.0 and R["expert__none"]["same"]["success"] == 1.0
    assert P["tasks"]["cross"]["n"] == 100 and P["settings"]["samples"] == 300 and P["settings"]["rounds"] == 10


def test_grid_reproduces_module_25():
    G = common.read_results("grid")
    m25 = json.loads((ROOT.parent / "25-world-models-and-planning" / "results" / "plan.json").read_text())
    assert G["arms"]["ema"]["plan_success"][0] == m25["agents"]["latent"]["success"]
    assert G["baselines"]["random"]["success"] == m25["agents"]["random"]["success"]
    assert G["baselines"]["perfect"]["success"] == m25["agents"]["perfect"]["success"]
    assert len(G["arms"]) == 8 and all(len(a["plan_success"]) == 3 for a in G["arms"].values())


def test_library_checks():
    L = common.read_results("library")
    assert L["world"]["max_position_diff"] == 0.0
    assert max(L["losses"][k] for k in L["losses"] if k.endswith("diff")) < 1e-5
    assert L["package"]["stable-worldmodel"] == "0.1.1"
    ours = common.read_results("plan")["rows"]["lewm_s0__cem"]
    for s in ("cross", "same"):
        assert L["plan"]["our_cem"][s]["success"] == ours[s]["success"]
        assert L["plan"]["same_outcome_tasks_noclip"][s] == 100            # the library's CEM, task by task


def test_followups_marked_and_reproduced():
    P = common.read_results("plan")
    assert P["followup_rows"]["written"].startswith("after the results")
    assert set(P["followup_rows"]) - {"written"} == {"lewm_s0__cem_best", "dim192_s0__cem_best", "lewm_s0__cem_noclip"}
    F = common.read_results("grid_followup")
    assert F["written"].startswith("after the results")
    for arm in F["arms"].values():
        assert arm["reproduced_plan_success"] == arm["saved_plan_success"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("name", ["world", "train", "plan", "grid"])
def test_figures_exist(name):
    assert (common.FIGURES / f"{name}.png").stat().st_size > 10_000
