"""Solution C — expected calibration error."""

import numpy as np


def ece(y_true, probability, n_bins=5):
    y = np.asarray(y_true)
    p = np.asarray(probability)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    result = 0.0
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if mask.any():
            result += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(result)
