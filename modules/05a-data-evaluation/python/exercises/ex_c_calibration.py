"""Exercise C — expected calibration error."""

import numpy as np


def ece(y_true, probability, n_bins=5):
    """Return example-weighted |confidence - frequency| across bins."""
    # TODO(you): bin probabilities, skip empty bins, weight by bin population.
    raise NotImplementedError("implement ECE")


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    calibrated = np.array([0.1, 0.3, 0.7, 0.9])
    overconfident = np.array([0.01, 0.05, 0.95, 0.99])
    print("calibrated:", ece(y, calibrated))
    print("overconfident:", ece(y, overconfident))
