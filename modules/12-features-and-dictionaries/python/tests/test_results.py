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

SCRIPTS = ("toy", "sae", "gpt2_sae", "steer")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def test_toy_sparsity_buys_room():
    r = common.read_results("toy")
    for name in ("A", "B"):
        f = r["setups"][name]["frames"]
        m = r["setups"][name]["m"]
        assert f[0]["n_represented"] == m                     # dense: one feature per dimension
        assert f[0]["n_represented_with_interference"] == 0
        assert f[-1]["n_represented"] > m                     # sparse: more features than dimensions


def test_sae_sweep_and_features():
    r = common.read_results("sae")
    sw = r["sweep"]
    assert [s["k"] for s in sw] == sorted(s["k"] for s in sw)
    assert all(abs(s["l0"] - s["k"]) < 0.5 for s in sw)      # TopK: L0 is k
    fves = [s["fve"] for s in sw]
    assert fves == sorted(fves)                               # more latents per input, better fit
    lm = r["lm_loss"]
    for s in sw:
        assert lm["intact"] < s["lm_loss_spliced"] < min(lm["mean_ablated"], lm["zero_ablated"])
    assert len(r["latents"]) == 24 and len(r["neurons"]) == 5
    for u in r["latents"] + r["neurons"]:
        assert len(u["contexts"]) == 10
        assert u["contexts"][0]["share_of_max"] == 1.0


def test_gpt2_sae_checks():
    r = common.read_results("gpt2_sae")
    assert r["mps_vs_cpu"]["same_top1_every_position"]
    assert r["mps_vs_cpu"]["max_abs_diff_logits"] < 1e-2
    s = r["stats"]
    assert s["d_sae"] == 24576 and s["tokens"] == 512 * 127
    assert s["loss_intact"] < s["loss_spliced"] < s["loss_zero_ablated"]
    assert all(f["url"].startswith("https://www.neuronpedia.org/gpt2-small/8-res-jb/") for f in r["features"])


def test_steer_conditions():
    r = common.read_results("steer")
    c = r["conditions"]
    assert set(c) == {"plain", "prompt", "steer x2", "steer x4", "random x4"}
    assert all(len(v["replies"]) == len(v["p_positive"]) == len(r["requests"]) for v in c.values())


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
