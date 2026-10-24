"""The problem: online bin packing, its instances, the packer and the exact score.

Items arrive one at a time; each must go into an open bin with room, or into a
new bin, and never moves again. A heuristic is a function
``priority(item, bins) -> array`` that scores every open bin that can take the
item; the packer puts the item in the bin with the highest score. If no open
bin has room, the packer opens a new one.

The score of a heuristic on a set of instances is its mean excess over the L1
lower bound, in percent: 100 * (bins used - L1) / L1, where
L1 = ceil(sum of sizes / capacity). No packing can use fewer than L1 bins, so
0% is a floor no heuristic can pass. Lower is better.
"""

from __future__ import annotations

import math

import numpy as np

# Instance sets. Seeds are fixed: the loop only ever sees "evolve"; "heldout"
# comes from the same distribution with other seeds; "shift" from another one.
SETS = {
    # Weibull sizes, shape 3, scale 45, rounded and clipped to [1, 100].
    "evolve": {"kind": "weibull", "capacity": 100, "n_items": 1000, "seeds": [0, 1, 2, 3]},
    "heldout": {"kind": "weibull", "capacity": 100, "n_items": 1000, "seeds": list(range(100, 110))},
    # Uniform integer sizes 20..100 in bins of 150.
    "shift": {"kind": "uniform", "capacity": 150, "n_items": 500, "seeds": list(range(200, 205))},
}


def make_instance(kind: str, capacity: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "weibull":
        sizes = np.rint(45.0 * rng.weibull(3.0, size=n_items))
        sizes = np.clip(sizes, 1, capacity)
    elif kind == "uniform":
        sizes = rng.integers(20, 101, size=n_items)
    else:
        raise ValueError(kind)
    return sizes.astype(np.int64)


def instances(name: str) -> list[tuple[np.ndarray, int]]:
    spec = SETS[name]
    return [(make_instance(spec["kind"], spec["capacity"], spec["n_items"], s), spec["capacity"])
            for s in spec["seeds"]]


def l1_bound(sizes: np.ndarray, capacity: int) -> int:
    return math.ceil(int(sizes.sum()) / capacity)


class InvalidHeuristic(Exception):
    """The heuristic returned something the packer cannot use."""


def pack(sizes: np.ndarray, capacity: int, priority) -> np.ndarray:
    """Pack online; return the remaining room of every bin opened.

    ``priority`` sees the item size and the remaining room of the open bins
    that can take it (a float array), and must return one finite score per
    bin. NaN or a wrong shape raises InvalidHeuristic; -inf and +inf are
    allowed, since argmax still picks one bin (the first on a tie).
    """
    room = np.empty(len(sizes), dtype=np.float64)   # at most one bin per item
    n_bins = 0
    for item in sizes:
        item = float(item)
        open_room = room[:n_bins]
        fits = np.nonzero(open_room >= item)[0]
        if len(fits) == 0:
            room[n_bins] = capacity - item
            n_bins += 1
            continue
        scores = priority(item, open_room[fits].copy())
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(fits),):
            raise InvalidHeuristic(f"bad scores: shape {scores.shape}, want ({len(fits)},)")
        if np.isnan(scores).any():
            raise InvalidHeuristic("a score is NaN")
        room[fits[int(np.argmax(scores))]] -= item
    return room[:n_bins].copy()


def check_packing(sizes: np.ndarray, capacity: int, room: np.ndarray) -> None:
    """The packing is valid: no bin over capacity, every item placed exactly once."""
    if np.any(room < -1e-9):
        raise InvalidHeuristic("a bin is over capacity")
    if not np.isclose(capacity * len(room) - room.sum(), float(sizes.sum())):
        raise InvalidHeuristic("placed sizes do not add up to the items")


def excess(bins: list[int], name: str) -> float:
    """Mean excess over L1 in percent, from bins used per instance of a set."""
    out = []
    for (sizes, capacity), b in zip(instances(name), bins):
        lb = l1_bound(sizes, capacity)
        out.append(100.0 * (b - lb) / lb)
    return float(np.mean(out))


# The two classic heuristics, written as priority functions.
FIRST_FIT = '''def priority(item, bins):
    """First fit: the earliest opened bin that has room."""
    return -np.arange(len(bins), dtype=float)
'''

BEST_FIT = '''def priority(item, bins):
    """Best fit: the bin that the item leaves with least room."""
    return -(bins - item)
'''


def first_fit(item, bins):
    return -np.arange(len(bins), dtype=float)


def best_fit(item, bins):
    return -(bins - item)


def bins_used(priority, name: str) -> list[int]:
    out = []
    for sizes, capacity in instances(name):
        room = pack(sizes, capacity, priority)
        check_packing(sizes, capacity, room)
        out.append(len(room))
    return out
