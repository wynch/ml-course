import numpy as np
import pytest

from src.metrics import (
    Confusion,
    classification_metrics,
    confusion_counts,
    duplicate_overlap,
    expected_calibration_error,
    metrics_by_slice,
    split_indices,
)


def test_confusion_and_metrics():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.8, 0.1]
    assert confusion_counts(y, p, 0.5) == Confusion(1, 1, 1, 1)
    metrics = classification_metrics(y, p, 0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_perfect_calibration_for_binary_certainties():
    assert expected_calibration_error([0, 1], [0.0, 1.0], n_bins=2) == 0


def test_splits_are_deterministic_disjoint_and_complete():
    first = split_indices(100, seed=4)
    second = split_indices(100, seed=4)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    combined = np.concatenate(first)
    assert len(np.unique(combined)) == 100
    assert set(combined) == set(range(100))


def test_overlap_and_slices():
    assert duplicate_overlap(["a", "b"], ["b", "c"]) == ["b"]
    sliced = metrics_by_slice([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.7], ["a", "a", "b", "b"])
    assert set(sliced) == {"a", "b"}
    assert sliced["a"]["accuracy"] == 1
