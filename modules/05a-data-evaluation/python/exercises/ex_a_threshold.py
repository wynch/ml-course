"""Exercise A — choose a threshold on validation, report once on test."""

from __future__ import annotations

import numpy as np

from src.metrics import classification_metrics


def best_f1_threshold(y_true, probability, candidates):
    """Return the candidate threshold with the highest F1.

    TODO(you): evaluate every candidate. On ties, keep the smaller threshold.
    """
    raise NotImplementedError("choose the validation threshold")


if __name__ == "__main__":
    validation_y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    validation_p = np.array([0.1, 0.45, 0.48, 0.61, 0.9, 0.3, 0.72, 0.55])
    threshold = best_f1_threshold(validation_y, validation_p, [0.3, 0.5, 0.7])
    print("chosen threshold:", threshold)
