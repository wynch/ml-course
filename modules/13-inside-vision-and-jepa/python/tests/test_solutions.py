"""The reference solutions, checked against the scripts and the committed results.

(a) loads InceptionV1 (27 MB, cached by Lucent) on the CPU; (b) reads JSON only;
(c) loads the 1.4 MB gridjepa encoder from artifacts/. All run in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)
import torch  # noqa: E402


# ---------------------------------------------------------------- (a) hook
def test_channel_hook_matches_script():
    import solution_a_channel_hook as sol
    import v_neuron

    model = v_neuron.load_inception(True, torch.device("cpu"))
    rng = np.random.default_rng(0)
    x = v_neuron.to_tensor([v_neuron.draw_v(rng)[0] for _ in range(4)])
    mine = sol.channel_spatial_max(model, x, 55)
    ref = v_neuron.channel_max(model, x, torch.device("cpu"))[:, v_neuron.CHANNELS.index(55)]
    assert np.allclose(mine.numpy(), ref, atol=1e-3)
    # The first four Vs of section 1, on the CPU, against the committed (MPS) values.
    committed = common.read_results("v_neuron_acts")["trained"]["V"][v_neuron.CHANNELS.index(55)][:4]
    assert np.allclose(mine.numpy(), committed, atol=0.1)
    assert not model.mixed4e_1x1_pre_relu_conv._forward_hooks       # hook removed


# --------------------------------------------------------- (b) gap and rank
def test_gap_and_rank_match_results():
    import solution_b_gap_rank as sol

    r = common.read_results("v_neuron")
    acts = sol.load_acts()
    for vehicle, key, rkey in (("car", "gap_by_channel", "rank_55"),
                               ("truck", "gap_trucks_by_channel", "rank_trucks_55")):
        g = sol.gap(acts, vehicle)
        ref = np.array([r["trained"][key][str(c)] for c in sol.CHANNELS])
        assert np.allclose(g, ref, atol=0.01)      # the JSON keeps activations to 0.1
        assert sol.rank(g, 55) == r["trained"][rkey]


# ------------------------------------------------- (c) BFS and the path probe
def test_bfs_toy_grids():
    import solution_c_path_probe as sol

    toy = np.array([[0, 0, 1, 6],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])
    assert sol.bfs_steps(toy, (0, 0)) == 7.0
    assert sol.bfs_steps(toy, (3, 0)) == 0.0
    walled = np.array([[0, 1, 6]])
    assert sol.bfs_steps(walled, (0, 0)) == float("inf")


def test_path_probe_matches_results():
    import solution_c_path_probe as sol
    from grid_encoder import encode, load_random, load_states, load_trained, split_by_episode

    r = common.read_results("jepa_probe")["probes"]
    st = load_states()
    raw = st["goal_dist"].astype(np.float64)
    y = np.where(np.isfinite(raw), np.minimum(raw, sol.CAP), sol.CAP).astype(np.float32)[:, None]
    tr, te = split_by_episode(st["ep"], seed=0)
    for name, enc in (("trained", load_trained()), ("random_0", load_random(0))):
        Z = encode(enc, st["windows"], st["scalars"])
        assert sol.ridge_r2(Z, y, tr, te) == pytest.approx(r[name]["path_r2"], abs=1e-4)
