"""The committed results/*.json: complete, inside the laptop budget, equal to the
measurements they copy, and in the explorable. No model is loaded here.
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

SCRIPTS = ("v_neuron", "feature_vis", "clip", "jepa_probe")


@pytest.mark.parametrize("name", SCRIPTS)
def test_guard_budget(name):
    g = common.read_results(name)["guard"]
    assert g["minutes"] < 5
    assert g["peak_rss_gb"] < 8 and g["peak_mps_driver_gb"] < 8
    assert g["mps_high_watermark_ratio"] == "0.5"


def _get(d: dict, path: str):
    for k in path.split("."):
        d = d[k]
    return d


def _pairs():
    """(module results file, source key, the frozen copy) for every copied number."""
    copy = json.loads((ROOT / "results" / "measured.json").read_text())
    for name, block in copy.items():
        for path, value in block.get("values", {}).items():
            yield name, path, value


@pytest.mark.parametrize("name,path,value", list(_pairs()))
def test_numbers_equal_the_measurement(name, path, value):
    """The module's own run gives the measurement's numbers, to the last digit."""
    assert _get(common.read_results(name), path) == value


def _source_files() -> dict[str, dict]:
    """The measurement's own JSON files, when this module sits in its source repository."""
    found = {}
    repo = ROOT.parents[2]
    # Later experiments (retests) write the same keys; the first in sorted order is the source.
    for f in sorted(repo.glob("*/experiments/*/metrics.json")):
        d = json.loads(f.read_text())
        if "gap_by_channel" in d.get("trained", {}):
            found.setdefault("v_neuron", d)
        elif "trained_minus_best_random" in d:
            found.setdefault("jepa_probe", d)
    for f in sorted(repo.glob("*/experiments/*/posthoc.json")):
        d = json.loads(f.read_text())
        if "many_arenas_120k_transitions" in d:
            found.setdefault("jepa_many_arenas", d)
    return found


def test_copy_equals_source():
    src = _source_files()
    if not src:
        pytest.skip("source measurement files not present (published copy of the course)")
    copy = json.loads((ROOT / "results" / "measured.json").read_text())
    for name, block in copy.items():
        if block.get("check_source") is False:
            continue
        assert name in src, name
        for path, value in block.get("values", {}).items():
            assert _get(src[name], path) == value, (name, path)
        for key, value in block.get("copied", {}).items():
            assert src[name][key] == value, (name, key)


def test_v_neuron_shapes():
    r = common.read_results("v_neuron")
    a = common.read_results("v_neuron_acts")
    assert r["channels"] == a["channels"] == list(range(45, 66))
    for s in r["sets"]:
        assert len(a["trained"][s]) == 21
        assert all(len(row) == 200 for row in a["trained"][s])
        assert len(a["random_55"][s]) == 200
        assert len(a["thumbs"][s]) == 10
    assert r["mps_vs_cpu_max_abs"] < 1e-2 * (r["activation_range_55"][1] - r["activation_range_55"][0])


def test_clip_probe_and_neurons():
    r = common.read_results("clip")
    p = r["probe"]
    assert len(p["accuracy_trained"]) == len(p["accuracy_random_weights"]) == 13
    assert r["n_train"] == 1000 and r["n_test"] == 500
    assert len(r["neurons"]) == 3 and all(len(n["top"]) == 8 for n in r["neurons"])


def test_jepa_controls_recorded():
    r = common.read_results("jepa_probe")
    assert set(r["probes"]) == {"trained", "random_0", "random_1", "random_2", "raw_window"}
    assert r["n_train"] + r["n_test"] == r["n_states"] == 10_000


def test_explorable_in_sync():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "src" / "export_explorable.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
