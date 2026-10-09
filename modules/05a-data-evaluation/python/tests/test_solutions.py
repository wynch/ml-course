import numpy as np

from solution_a_threshold import best_f1_threshold
from solution_b_leakage import overlapping_ids
from solution_c_calibration import ece


def test_threshold_solution():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.45, 0.48, 0.61, 0.9, 0.3, 0.72, 0.55])
    assert best_f1_threshold(y, p, [0.3, 0.5, 0.7]) == 0.5


def test_leakage_solution():
    assert overlapping_ids(["b", "a", "b"], ["c", "b"]) == ["b"]


def test_calibration_solution():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0]), 5) == 0
