import numpy as np
import pytest

from src.metrics import (
    Confusion,
    bootstrap_ci,
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


def test_bootstrap_ci_brackets_the_point_and_narrows_with_more_data():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 400), 0, 1)
    point, low, high = bootstrap_ci(y[:40], p[:40], "accuracy", n_resamples=500)
    assert low <= point <= high
    _, low_big, high_big = bootstrap_ci(y, p, "accuracy", n_resamples=500)
    assert high_big - low_big < high - low
    assert bootstrap_ci(y[:40], p[:40], n_resamples=500) == bootstrap_ci(
        y[:40], p[:40], n_resamples=500
    )
