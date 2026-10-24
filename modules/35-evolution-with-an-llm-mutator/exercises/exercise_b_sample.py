"""Exercise (b): choosing a parent by score.

The database groups an island's programs into clusters that behave alike and
draws a cluster with probability proportional to exp(score / T), where score
is minus the excess in percentage points and T is a temperature.

Predict first: two clusters at 4.781% and 5.340% excess, T = 0.25. What
share of draws goes to the better one? And at T = 2.5?

Run the test: uv run pytest tests/test_solutions.py -k sample
"""

from __future__ import annotations

import numpy as np


def cluster_probs(excess: list[float], temperature: float) -> np.ndarray:
    """Probability of drawing each cluster. temperature = inf means uniform."""
    # TODO(you): subtract the largest score before exp(), so nothing overflows.
    raise NotImplementedError
