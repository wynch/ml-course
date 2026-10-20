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
import train as Tr  # noqa: E402

GB_CAP = 4.0          # the module's peak-memory guard


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_every_process_inside_the_guard():
    guards = _jsonl(common.RESULTS / "guards.jsonl")
    assert len(guards) >= len(Tr.ARMS)
    for g in guards:
        assert g["minutes"] < 5, g["script"]
        assert g["peak_rss_gb"] + g["peak_mps_driver_gb"] < GB_CAP, g["script"]
        assert g["mps_high_watermark_ratio"] == "0.3"
    for name in ("flow2d", "evaluate", "sample", "block", "library"):
        g = common.read_results(name)["guard"]
        assert g["minutes"] < 5 and g["peak_rss_gb"] + g["peak_mps_driver_gb"] < GB_CAP, name


def test_expectations_written_and_complete():
    E = common.read_results("expectations")
    assert E["written"].startswith("2026-10-20, before any run")
    assert list(E["expectations"]) == [f"E{i}" for i in range(1, 29)]


@pytest.mark.parametrize("arm", list(Tr.ARMS))
def test_every_arm_finished(arm):
    r = common.read_results(f"runs/{arm}")
    assert r["progress"]["step"] == Tr.spec(arm, 1_003_855)["steps"]
    assert r["params"] == 619_904
    if arm.startswith("r"):
        assert sorted(int(e) for e in r["marks"]) == sorted(r["spec"]["marks"])


def test_likelihood_numbers_hang_together():
    L = common.read_results("evaluate")["likelihood_nats_per_char"]
    assert L["ar_exact"] < L["mdm_elbo"] < L["unigram"]
    assert abs(L["mdm_elbo"] - L["mdm_elbo_seed1"]) < 0.02          # the bound's Monte Carlo noise is small
    for k in ("mdm_exact_l2r", "mdm_exact_r2l", "mdm_exact_random", "bd4_elbo", "bd16_elbo"):
        assert 1.0 < L[k] < 4.2, k
    assert abs(common.read_results("runs/ar")["val_final"]["nll"] - L["ar_exact"]) < 1e-3


def test_sampling_tables_complete():
    S = common.read_results("sample")
    for m in ("mdm", "mdm6x"):
        for rule in ("random", "confidence"):
            assert set(S["results"][m][rule]) == {str(n) for n in S["steps"]}
    T = common.read_results("traces")["traces"]
    assert len(T) == 2 * 2 * len(S["steps"]) * 4
    for t in T:
        assert len(t["text"]) == 128 and min(t["when"]) == 1 and max(t["when"]) == t["steps"]


def test_block_cache_and_costs():
    B = common.read_results("block")["results"]
    for tot in ("128", "32"):
        for name in ("bd4", "bd16"):
            assert B[tot][f"{name}_cache"]["same_chars_as_nocache"] >= 0.99
            assert B[tot][f"{name}_cache"]["positions"] < B[tot][f"{name}_nocache"]["positions"]
    assert B["128"]["mdm"]["positions"] == 128 * 128


def test_library_checks():
    Lb = common.read_results("library")
    assert Lb["continuous"]["x_t_max_gap"] < 1e-6 and Lb["continuous"]["velocity_max_gap"] < 1e-6
    assert Lb["continuous"]["euler16_max_gap"] < 1e-5
    assert Lb["loss"]["max_relative_gap"] < 1e-5
    assert Lb["llada"]["same_as_ours_16_steps"] == 1.0 and Lb["llada"]["same_as_ours_128_steps"] == 1.0


def test_repeat_summary_matches_runs():
    R = common.read_results("repeat")["sizes"]
    assert R["r100k"]["unique_chars"] == 100_000 and len(R["r100k"]["mdm"]["seeds"]) == 3
    run = common.read_results("runs/r100k_ar_s0")
    e = str(R["r100k"]["ar"]["epochs"][-1])
    vals = [common.read_results(f"runs/r100k_ar_s{s}")["marks"][e]["val"]["nll"] for s in (0, 1, 2)]
    assert abs(sum(vals) / 3 - R["r100k"]["ar"]["val_mean"][-1]) < 1e-3
    assert run["spec"]["sched"] == "constant"


def test_artifact_present():
    assert (common.ARTIFACTS / "models.pt").stat().st_size < 7_000_000


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
