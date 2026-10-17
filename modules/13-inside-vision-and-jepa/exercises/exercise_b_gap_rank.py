"""Exercise (b): the gap and the rank, from the committed activations. No model.

results/v_neuron_acts.json holds, for each of the six sets, the activation of
channels 45 to 65 of mixed4e on all 200 images. Write ``gap``, the number
section 1 ranks the channels by, and ``rank``.

Predict before you run: of the 21 channels, where does 4e:55 rank? Then swap
the CIFAR cars for the garbage trucks (``vehicle="truck"``): does it move?

    uv run python ../exercises/exercise_b_gap_rank.py
    uv run pytest tests/test_solutions.py -k gap -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

import numpy as np

CHANNELS = list(range(45, 66))


def load_acts() -> dict[str, np.ndarray]:
    """Set name -> (21, 200) activations of channels 45..65, from results/v_neuron_acts.json."""
    r = json.loads((ROOT / "results" / "v_neuron_acts.json").read_text())
    return {s: np.array(v) for s, v in r["trained"].items()}


def gap(acts: dict[str, np.ndarray], vehicle: str = "car") -> np.ndarray:
    """(median V - max(median cat, median vehicle)) / pooled SD, one number per channel.

    Pooled SD = sqrt of the mean of the three sets' variances (ddof = 1).
    """
    # TODO(you): medians of "V", "cat" and `vehicle` along axis 1, the pooled
    # SD of those three sets, and the gap. Return shape (21,).
    ...


def rank(g: np.ndarray, channel: int) -> int:
    """1 = the largest gap of the 21."""
    # TODO(you): count the channels with a larger gap than `channel`, plus one.
    ...


def main() -> None:
    acts = load_acts()
    g = gap(acts)
    print(f"4e:55 gap {g[CHANNELS.index(55)]:+.2f} SD, rank {rank(g, 55)} of {len(CHANNELS)}")
    print(f"largest gap: 4e:{CHANNELS[int(np.argmax(g))]} at {g.max():+.2f} SD")


if __name__ == "__main__":
    main()
