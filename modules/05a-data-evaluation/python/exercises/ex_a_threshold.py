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

    # Look at the test split once, after the rule is frozen.
    test_y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    test_p = np.array([0.66, 0.52, 0.41, 0.2, 0.35, 0.83, 0.58, 0.47])
    m = classification_metrics(test_y, test_p, threshold)
    print(f"test at {threshold}: precision={m['precision']:.3f}, "
          f"recall={m['recall']:.3f}, f1={m['f1']:.3f}")
