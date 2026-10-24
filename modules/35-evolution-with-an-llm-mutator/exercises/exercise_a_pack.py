"""Exercise (a): the online packer and the L1 bound.

Predict first: on the evolve set (4 instances of 1,000 Weibull items, bins of
100), how many points over the L1 bound is best fit? And first fit?

Run the test: uv run pytest tests/test_solutions.py -k pack
"""

from __future__ import annotations

import math

import numpy as np


def l1_bound(sizes: np.ndarray, capacity: int) -> int:
    # TODO(you): the fewest bins any packing could use, from the total size alone.
    raise NotImplementedError


def pack(sizes: np.ndarray, capacity: int, priority) -> int:
    """Pack items in order; return the number of bins used.

    For each item, call priority(item, room) with the remaining room of the
    open bins that can take it (a float array), and put the item in the bin
    with the highest score. If no open bin has room, open a new one.
    """
    # TODO(you): keep an array of remaining room, one entry per open bin.
    raise NotImplementedError


def excess(sizes: np.ndarray, capacity: int, bins: int) -> float:
    """Percent over the L1 bound."""
    lb = l1_bound(sizes, capacity)
    return 100.0 * (bins - lb) / lb


_ = math  # math.ceil may help
