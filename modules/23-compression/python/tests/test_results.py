"""The committed results/*.json: complete, inside the laptop budget, and in the explorable.

No model is loaded here; these run in a second.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import common  # noqa: E402

SCRIPTS = ("distil", "prune", "quant", "mlx_way")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_distil_pairs():
    r = common.read_results("distil")
    assert 0.45 < r["student"]["share_of_teacher"] < 0.55
    seeds = r["setup"]["seeds"]
    assert len(r["runs"]) == 2 * len(seeds)
    for run in r["runs"]:
        assert run["curve"][-1]["step"] == r["setup"]["steps"]
        assert run["val_loss"] > r["teacher"]["val_loss"]          # the student stays behind its teacher
    assert len(r["gap_scratch_minus_distil"]) == len(seeds)


def test_prune_rows():
    r = common.read_results("prune")
    rows = r["rows"]
    assert rows[0]["method"] == "untouched" and rows[0]["fraction"] == 0.0
    assert rows[0]["tokens"] == common.PPL_WINDOWS * (common.PPL_LEN - 1)
    for row in rows[1:]:
        assert abs(row["zero_share_of_block_weights"] - row["fraction"]) < 0.002
        assert row["tokens"] == rows[0]["tokens"]
    assert r["mps_vs_cpu"]["same_top1_every_position"]
    assert r["mps_vs_cpu"]["max_abs_diff_logits"] < 1e-2


def test_quant_rows():
    r = common.read_results("quant")
    rows = r["rows"]
    assert rows[0]["scheme"].startswith("untouched") and rows[0]["mean_rel_error"] == 0
    assert rows[0]["bytes"] == 2 * r["all_params"]
    by = {row["scheme"]: row for row in rows}
    assert by["4-bit, group 64"]["bits_per_weight"] == 4.5
    assert abs(by["int8, per row"]["bits_per_weight"] - 8) < 0.05
    assert by["4-bit, group 64"]["mean_rel_error"] < by["4-bit, per row"]["mean_rel_error"]


def test_mlx_matches_hand():
    m = common.read_results("mlx_way")
    q = common.read_results("quant")
    c = m["rows"][1]["check_vs_hand"]
    assert c["matrices"] == 225 and c["bit_identical"] == c["matrices"]   # 224 block matrices + embedding
    assert m["ppl_tokens"] == q["rows"][0]["tokens"]
    hand_g64 = next(r for r in q["rows"] if r["scheme"] == "4-bit, group 64")
    # The stored file holds the same integers, scales and biases the hand count assumes, plus headers.
    assert 0 < m["safetensors_bytes"]["4bit_g64"] - hand_g64["bytes"] < 1e6


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
