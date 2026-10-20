"""Exercise C — expected calibration error."""

import numpy as np


def ece(y_true, probability, n_bins=5):
    """Return example-weighted |confidence - frequency| across bins."""
    # TODO(you): bin probabilities, skip empty bins, weight by bin population.
    raise NotImplementedError("implement ECE")


if __name__ == "__main__":
    # Four examples: each bin's observed rate is 0 or 1, so near-certainty wins.
    y = np.array([0, 0, 1, 1])
    print("tiny set, 'honest' :", ece(y, np.array([0.1, 0.3, 0.7, 0.9])))
    print("tiny set, 'certain':", ece(y, np.array([0.01, 0.05, 0.95, 0.99])))
    # Ten examples, one mistake per side: now the names hold.
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    print("calibrated   :", ece(y, np.array([0.2] * 5 + [0.8] * 5)))
    print("overconfident:", ece(y, np.array([0.02] * 5 + [0.98] * 5)))
