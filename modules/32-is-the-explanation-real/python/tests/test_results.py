"""The committed results/*.json: complete, inside the laptop budget, consistent, and in the explorable.

Nothing is trained here; these run in about a second.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

BUDGET_GB = 4.0


def chunks(name):
    return [json.loads(line) for line in (common.RESULTS / f"{name}_log.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("log", ["train_sparse", "train_dense", "dicts", "library"])
def test_every_chunk_under_five_minutes_and_budget(log):
    rows = chunks(log)
    assert rows
    for r in rows:
        assert r["seconds"] < 300
        if r["peak_rss_gb"] is None:            # one chunk that stopped at the MPS cap; its note says so
            assert "out-of-memory" in r["note"]
            continue
        assert r["peak_rss_gb"] < BUDGET_GB and r["peak_mps_driver_gb"] < BUDGET_GB


@pytest.mark.parametrize("name", ["dicts", "attribution", "library"])
def test_analysis_scripts_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5 and g["peak_rss_gb"] < BUDGET_GB and g["peak_mps_driver_gb"] < BUDGET_GB
    assert g["mps_high_watermark_ratio"] == "0.3"


def test_expectations_written_first():
    E = json.loads((common.RESULTS / "expectations.json").read_text())
    assert E["written"].startswith("2026-10-20, before any training run")
    assert set(E["expectations"]) == {"T1", "T2", "G1", "G2", "G3", "R1", "R2", "R3", "S1", "S2",
                                      "A1", "A2", "A3", "A4", "L1", "L2"}


def test_training():
    S, D = common.read_results("train_sparse"), common.read_results("train_dense")
    assert S["config"]["k"] == 8 and S["config"]["n_mlp"] == 2048 and D["config"]["k"] == 0
    assert S["steps"] == D["steps"] == 8000
    assert S["chunks_log"] == "results/train_sparse_log.jsonl"
    assert chunks("train_sparse")[-1]["steps"][1] == 8000 and chunks("train_dense")[-1]["steps"][1] == 8000


def test_dicts_consistent():
    R = common.read_results("dicts")
    for which in ("trained", "random"):
        M = R["models"][which]
        for kind in ("sae", "transcoder"):
            K = M["dicts"][kind]
            vals = [K[f"s{s}"]["mmcs_value"] for s in (0, 1, 2)]
            assert K["mmcs_mean"] == pytest.approx(np.mean(vals), abs=1e-4)
            assert K["mmcs_spread"] == pytest.approx(np.ptp(vals), abs=1e-4)
            for s in (0, 1, 2):
                assert 0 <= K[f"s{s}"]["recovered_share"] <= 1
                assert sum(K[f"s{s}"]["best_cos_hist"]) == M["true_alive"]
            assert set(K["seeds"]["shared_share_pairs"]) == {"0-1", "0-2", "1-2"}
    assert R["n_eval_tokens"] == 2**18


def test_attribution_consistent():
    A = common.read_results("attribution")
    assert A["n_prompts"] == 20 and len(A["prompts_text"]) == 20
    for M in A["models"].values():
        assert len(M["prompts"]) == 20
        assert M["frozen_check_max_gap"] < 1e-3                 # attribution is exact in the frozen model
        assert len(M["scatter"]["att"]) == sum(r["n_active"] for r in M["prompts"])
        assert 0 <= M["top10_beats_random_mean"] <= 20


def test_library():
    L = common.read_results("library")
    assert set(L["saes"]) == {"trained", "random", "matched", "published"}
    for k in ("trained", "random", "matched"):
        assert L["saes"][k]["l0"] == pytest.approx(32, abs=0.01)
        assert L["saes"][k]["guard_train"]["minutes"] < 5


def test_artifacts():
    assert (common.ARTIFACTS / "sparse.pt").stat().st_size < 4_000_000
    assert (common.ARTIFACTS / "dense.pt").stat().st_size < 2_500_000
    assert (common.ARTIFACTS / "vocab.json").exists()


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
