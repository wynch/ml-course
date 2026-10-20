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

ARMS = ["full", "full_slow", "lora4", "lora32", "replay", "distill"]
PLAST = ["backprop", "cbp", "sgd", "fresh"]
LIB = ["lora8_gentle", "lora8", "lora64", "lora8_replay"]
SEEDS = [0, 1, 2]


def run(folder, name):
    return json.loads((common.RESULTS / folder / f"{name}.json").read_text())


def guards():
    yield "start", common.read_results("start")["guard"]
    for a in ARMS:
        for s in SEEDS:
            yield f"{a}_s{s}", run("runs", f"{a}_s{s}")["guard"]
    yield "library_memory", common.read_results("library_memory")["guard"]


@pytest.mark.parametrize("name,g", list(guards()))
def test_guard_budget(name, g):
    assert g["minutes"] < 5, name                      # every script or chunk under five minutes
    assert g["peak_rss_gb"] + g["peak_mps_driver_gb"] < 4, name
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_long_jobs_budget():
    S = common.read_results("start")
    assert S["guard"]["peak_rss_gb_all_chunks"] < 4
    for a in PLAST:
        for s in SEEDS:
            assert run("plasticity", f"{a}_s{s}")["peak_rss_gb"] < 4
    for a in LIB:
        r = run("library", a)
        assert r["peak_rss_gb"] + r["peak_mps_driver_gb"] < 4, a
    # the first memory measurement, batch 4 at once, went over the guard; it is kept and quoted, not repeated
    assert common.read_results("library_memory_batch4")["guard"]["peak_mps_driver_gb"] > 4


def test_all_runs_present():
    for a in ARMS:
        for s in SEEDS:
            r = run("runs", f"{a}_s{s}")
            assert r["arm"] == a and r["seed"] == s and r["steps"] == 600
            assert [row["step"] for row in r["log"]] == list(range(0, 601, 25))
            assert r["log"][0]["kl_new"] == pytest.approx(0, abs=1e-6)
            assert r["final"]["forgetting"] == pytest.approx(r["log"][-1]["old_loss"] - r["log"][0]["old_loss"], abs=1e-4)
    for a in PLAST:
        for s in SEEDS:
            r = run("plasticity", f"{a}_s{s}")
            assert len(r["log"]) == (40 if a == "fresh" else 800)
    for a in LIB:
        assert [row["step"] for row in run("library", a)["log"]] == [0, 50, 100, 200, 300]


def test_start_model():
    S = common.read_results("start")
    assert S["parameters"] == 623_104 and S["vocab_size"] == 91
    assert S["final"]["old_loss"] < 1.8 and S["final"]["new_loss"] > 3
    assert (common.ARTIFACTS / "start.pt").stat().st_size < 3_000_000
    for a in ARMS:                                     # every arm starts from the same model
        assert run("runs", f"{a}_s0")["log"][0]["old_loss"] == pytest.approx(S["final"]["old_loss"], abs=1e-4)


def test_summaries_match_runs():
    F = common.read_results("forget")
    for a in ARMS:
        assert F["arms"][a]["final"]["forgetting"]["seeds"] == pytest.approx(
            [run("runs", f"{a}_s{s}")["final"]["forgetting"] for s in SEEDS], abs=1e-4)
    assert set(F["expectations_met"]) == {f"E{i}" for i in range(1, 9)}
    P = common.read_results("plasticity")
    assert set(P["expectations_met"]) == {f"P{i}" for i in range(1, 6)}
    L = common.read_results("library")
    assert set(L["expectations_met"]) == {f"L{i}" for i in range(1, 5)}
    for a in LIB:
        assert L["arms"][a]["final"] == run("library", a)["final"]
    exp = common.read_results("expectations")["expectations"]
    for R in (F, P, L):
        for k, v in R["expectations_met"].items():
            assert v["text"] == exp[k]


def test_library_lora_starts_at_the_base():
    for a in LIB:
        first = run("library", a)["log"][0]
        assert first["kl_new"] == pytest.approx(0, abs=1e-3) and first["kl_probe"] == pytest.approx(0, abs=1e-3)


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
