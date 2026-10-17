"""Reference solution, exercise (a): read one InceptionV1 channel with a hook."""

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
        seen.append(output[:, channel].amax(dim=(1, 2)))

    handle = model.mixed4e_1x1_pre_relu_conv.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x * 255 - 117)
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
