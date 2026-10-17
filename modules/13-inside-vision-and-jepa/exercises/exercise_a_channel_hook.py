"""Exercise (a): read one InceptionV1 channel with a forward hook.

``channel_spatial_max`` should register a forward hook on
``model.mixed4e_1x1_pre_relu_conv``, keep the largest value of one channel over
the 14 x 14 map for each image, run the model, and remove the hook.
InceptionV1 (the Lucent copy of the TensorFlow weights) wants pixels as
x * 255 - 117, not in [0, 1].

Predict before you run: on 8 new drawn Vs and 8 new drawn Os, which set gets
the higher median for 4e:55? Section 1 says what 200 of each gave.

    uv run python ../exercises/exercise_a_channel_hook.py
    uv run pytest tests/test_solutions.py -k hook -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import numpy as np
import torch

from v_neuron import draw_o, draw_v, load_inception, to_tensor

CHANNEL = 55


def channel_spatial_max(model, x: torch.Tensor, channel: int) -> torch.Tensor:
    """x (N, 3, 224, 224) in [0, 1] -> (N,): max over the map of mixed4e:channel before ReLU."""
    seen = []

    def hook(module, inputs, output):
        # TODO(you): output is (N, 256, 14, 14). Append, for `channel`, the max
        # over the two spatial dimensions, shape (N,), to `seen`.
        ...

    # TODO(you): register `hook` as a forward hook on model.mixed4e_1x1_pre_relu_conv.
    handle = ...
    try:
        with torch.no_grad():
            # TODO(you): run the model on x, rescaled to InceptionV1's input range.
            ...
    finally:
        handle.remove()
    return seen[0]


def main() -> None:
    model = load_inception(True, torch.device("cpu"))
    rng = np.random.default_rng(1)
    vs = to_tensor([draw_v(rng)[0] for _ in range(8)])
    os_ = to_tensor([draw_o(rng)[0] for _ in range(8)])
    print(f"4e:{CHANNEL} on 8 new Vs: median {float(channel_spatial_max(model, vs, CHANNEL).median()):.1f}")
    print(f"4e:{CHANNEL} on 8 new Os: median {float(channel_spatial_max(model, os_, CHANNEL).median()):.1f}")


if __name__ == "__main__":
    main()
