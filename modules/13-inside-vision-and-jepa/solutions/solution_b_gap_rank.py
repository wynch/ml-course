"""Reference solution, exercise (b): the gap and the rank, from the committed activations."""

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
    med = {k: np.median(acts[k], axis=1) for k in ("V", "cat", vehicle)}
    var = np.mean([acts[k].var(axis=1, ddof=1) for k in ("V", "cat", vehicle)], axis=0)
    return (med["V"] - np.maximum(med["cat"], med[vehicle])) / np.sqrt(var)


def rank(g: np.ndarray, channel: int) -> int:
    """1 = the largest gap of the 21."""
    return int((g > g[CHANNELS.index(channel)]).sum()) + 1


def main() -> None:
    acts = load_acts()
    g = gap(acts)
    print(f"4e:55 gap {g[CHANNELS.index(55)]:+.2f} SD, rank {rank(g, 55)} of {len(CHANNELS)}")
    print(f"largest gap: 4e:{CHANNELS[int(np.argmax(g))]} at {g.max():+.2f} SD")


if __name__ == "__main__":
    main()
