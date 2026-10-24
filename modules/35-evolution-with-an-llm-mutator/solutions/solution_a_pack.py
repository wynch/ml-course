"""Solution (a): the online packer and the L1 bound."""

from __future__ import annotations

import math

import numpy as np


def l1_bound(sizes: np.ndarray, capacity: int) -> int:
    return math.ceil(int(np.sum(sizes)) / capacity)


def pack(sizes: np.ndarray, capacity: int, priority) -> int:
    room: list[float] = []
    for item in sizes:
        item = float(item)
        fits = [i for i, r in enumerate(room) if r >= item]
        if not fits:
            room.append(capacity - item)
            continue
        scores = np.asarray(priority(item, np.array([room[i] for i in fits])), dtype=float)
        room[fits[int(np.argmax(scores))]] -= item
    return len(room)


def excess(sizes: np.ndarray, capacity: int, bins: int) -> float:
    lb = l1_bound(sizes, capacity)
    return 100.0 * (bins - lb) / lb
