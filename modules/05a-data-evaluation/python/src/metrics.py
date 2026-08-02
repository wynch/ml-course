"""Evaluation primitives implemented from scratch with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def _as_arrays(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(probability, dtype=np.float64)
    if truth.ndim != 1 or prob.ndim != 1 or truth.shape != prob.shape:
        raise ValueError("y_true and probability must be same-length 1D arrays")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return truth, prob


def confusion_counts(y_true, probability, threshold: float = 0.5) -> Confusion:
    """Return TP/FP/FN/TN counts at ``threshold``."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    truth, prob = _as_arrays(y_true, probability)
    pred = prob >= threshold
    positive = truth == 1
    return Confusion(
        tp=int(np.sum(pred & positive)),
        fp=int(np.sum(pred & ~positive)),
        fn=int(np.sum(~pred & positive)),
        tn=int(np.sum(~pred & ~positive)),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    y_true, probability, threshold: float = 0.5
) -> dict[str, float]:
    """Compute common binary metrics from a confusion matrix."""
    c = confusion_counts(y_true, probability, threshold)
    precision = _safe_div(c.tp, c.tp + c.fp)
    recall = _safe_div(c.tp, c.tp + c.fn)
    return {
        "accuracy": _safe_div(c.tp + c.tn, c.total),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(c.tn, c.tn + c.fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "tp": float(c.tp),
        "fp": float(c.fp),
        "fn": float(c.fn),
        "tn": float(c.tn),
    }


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    """Example-weighted calibration gap over equal-width probability bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    truth, prob = _as_arrays(y_true, probability)
    if len(truth) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(prob, edges[1:-1]), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = float(prob[mask].mean())
        frequency = float(truth[mask].mean())
        error += float(mask.mean()) * abs(confidence - frequency)
    return error


def split_indices(
    n_examples: int,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic, disjoint train/validation/test row indices."""
    if n_examples < 3:
        raise ValueError("need at least three examples")
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train and validation ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must leave room for a test split")

    order = np.random.default_rng(seed).permutation(n_examples)
    n_train = int(n_examples * train_ratio)
    n_validation = int(n_examples * validation_ratio)
    return (
        order[:n_train],
        order[n_train : n_train + n_validation],
        order[n_train + n_validation :],
    )


def duplicate_overlap(train_ids, test_ids) -> list[str]:
    """Return sorted exact IDs that occur in both splits."""
    return sorted(set(map(str, train_ids)) & set(map(str, test_ids)))


def metrics_by_slice(
    y_true, probability, slices, threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Compute metrics for every distinct slice label."""
    truth, prob = _as_arrays(y_true, probability)
    labels = np.asarray(slices)
    if labels.shape != truth.shape:
        raise ValueError("slices must have one label per example")
    return {
        str(label): classification_metrics(truth[labels == label], prob[labels == label], threshold)
        for label in np.unique(labels)
    }
