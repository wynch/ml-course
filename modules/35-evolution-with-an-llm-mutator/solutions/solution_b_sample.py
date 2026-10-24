"""Solution (b): choosing a parent by score."""

from __future__ import annotations

import math

import numpy as np


def cluster_probs(excess: list[float], temperature: float) -> np.ndarray:
    scores = -np.asarray(excess, dtype=float)
    if math.isinf(temperature):
        w = np.ones(len(scores))
    else:
        w = np.exp((scores - scores.max()) / temperature)
    return w / w.sum()
