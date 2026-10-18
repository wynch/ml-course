"""Exercise (b): the shuffled-y control, on the committed nets.

Give each prompt the y values of another prompt, so no y can be predicted from its own
x. Then score two nets: the net trained on real prompts, and the control net trained on
shuffled ones (both in artifacts/, written by src/train_icl.py).

1. ``shuffle_y``: prompt b gets the y values of prompt b+1.
2. ``score``: MSE/d at each n.

Predict before you run: guessing 0 scores 1. What does the control net score on
shuffled prompts? And the main net, which has only seen prompts that hold a rule?
Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_b_shuffled.py
    uv run pytest tests/test_solutions.py -k shuffled -v
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
    # TODO(you): one line (np.roll along the prompt axis).
    ...


def load(name: str) -> icl.ICLNet:
    net = icl.ICLNet()
    net.load_state_dict(torch.load(common.ARTIFACTS / f"{name}.pt"))
    return net.eval()


def score(net: icl.ICLNet, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """MSE/d at each n = 0..20."""
    pred = icl.net_predict(net, xs, ys, torch.device("cpu"))
    # TODO(you): mean over prompts of (pred - ys)^2, divided by d.
    ...


def main() -> None:
    xs, ys, _ = icl.eval_set(seed=11, shuffled=False)
    ys_s = shuffle_y(ys)
    main_net, ctrl = load("icl_net"), load("icl_control")
    for name, net in (("main net", main_net), ("control net", ctrl)):
        real, shuf = score(net, xs, ys), score(net, xs, ys_s)
        print(f"{name:12s} real prompts, n = 10: {real[10]:.3f}   shuffled, mean over n: {shuf.mean():.3f}")


if __name__ == "__main__":
    main()
