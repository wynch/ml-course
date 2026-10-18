"""The committed results: complete, inside the laptop budget, and in the explorable.

No model is loaded; these run in a few seconds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import analyse  # noqa: E402
import common  # noqa: E402


def test_generations_complete():
    c = analyse.load()
    assert len(c["ids"]) == 200
    assert all(len(c["samples"][i]["samples"]) == 8 for i in c["ids"])
    assert all(len(c["verify"][i]["verdicts"]) == 8 for i in c["ids"])


def test_analysis_matches_generations():
    """Every number in analysis.json comes from the cached generations."""
    fresh = analyse.analyse(analyse.load())
    saved = common.read_results("analysis")
    for key in ("section1", "section2", "section3", "cost", "per_problem"):
        a = __import__("json").loads(__import__("json").dumps(fresh[key]))
        assert a == saved[key], key


def test_curves_are_sane():
    a = common.read_results("analysis")
    s2, s3 = a["section2"], a["section3"]
    p = s3["pass_at_n_mean_over_subsets"]
    assert all(p[str(n)] <= p[str(n + 1)] + 1e-12 for n in range(1, 8))     # any-of-n only grows
    for n in range(1, 9):
        assert s2["vote_mean_over_subsets"][str(n)] <= p[str(n)] + 1e-12    # a vote cannot beat the ceiling
        assert s3["best_mean_over_subsets"][str(n)] <= p[str(n)] + 1e-12
    assert abs(s2["vote_mean_over_subsets"]["1"] - s2["single_sample_mean"]) < 1e-12


def test_guard_budget():
    for name in ("analysis", "device_check"):
        g = common.read_results(name)["guard"]
        assert g["minutes"] < 5
        assert g["peak_rss_gb"] < 8 and g["peak_metal_gb"] < 8


def test_device_check():
    """In float32 the devices agree; in bfloat16 the CPU's 4-bit kernel drifts (see the README)."""
    d = common.read_results("device_check")
    assert d["tokens"] == 32
    assert d["gpu_vs_cpu_f32"]["top1_agree_positions"] == d["tokens"]
    assert d["gpu_bf16_vs_gpu_f32"]["top1_agree_positions"] == d["tokens"]
    layer = d["one_layer_max_error_vs_float64"]
    assert abs(layer["cpu_qmm_bf16"] - layer["bf16_running_sum_numpy"]) < 1e-4
    assert layer["gpu_qmm_bf16"] < layer["cpu_qmm_bf16"]


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
