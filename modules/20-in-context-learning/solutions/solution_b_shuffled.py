"""Solution (b): the shuffled-y control, on the committed nets.

Shuffle the y values against the x values and score both nets: the net trained on
real prompts, and the control net trained on shuffled ones.

    uv run python ../solutions/solution_b_shuffled.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np
import torch

import icl


def shuffle_y(ys: np.ndarray) -> np.ndarray:
    """Prompt b gets the y values of prompt b+1 (the last gets the first's)."""
    return np.roll(ys, -1, axis=0)


def load(name: str) -> icl.ICLNet:
    net = icl.ICLNet()
    net.load_state_dict(torch.load(common.ARTIFACTS / f"{name}.pt"))
    return net.eval()


def score(net: icl.ICLNet, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """MSE/d at each n = 0..20."""
    pred = icl.net_predict(net, xs, ys, torch.device("cpu"))
    return ((pred - ys) ** 2).mean(0) / icl.D


def main() -> None:
    xs, ys, _ = icl.eval_set(seed=11, shuffled=False)
    ys_s = shuffle_y(ys)
    main_net, ctrl = load("icl_net"), load("icl_control")
    for name, net in (("main net", main_net), ("control net", ctrl)):
        real, shuf = score(net, xs, ys), score(net, xs, ys_s)
        print(f"{name:12s} real prompts, n = 10: {real[10]:.3f}   shuffled, mean over n: {shuf.mean():.3f}")


if __name__ == "__main__":
    main()
