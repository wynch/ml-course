"""The program database: islands of programs, grouped by behaviour, sampled by score.

Each island evolves on its own. Inside an island, programs that use the same
number of bins on every evolve instance form one cluster (their behaviour
signature). A parent is drawn in two steps: a cluster, with probability
proportional to exp(score / T), where score = -excess in percentage points;
then a program inside it, shorter ones more likely. Every RESET_EVERY steps
the worse half of the islands is wiped and reseeded with the best program of
a random surviving island.

The same scheme, at a far larger scale, is FunSearch's (Romera-Paredes et
al., 2024, Methods); the numbers here are ours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Program:
    id: int
    code: str
    excess: float                 # percent over L1 on the evolve set; lower is better
    signature: tuple
    length: int                   # characters of normalized code

    @property
    def score(self) -> float:
        return -self.excess


@dataclass
class Island:
    clusters: dict = field(default_factory=dict)   # signature -> list[Program]

    def add(self, p: Program) -> None:
        self.clusters.setdefault(p.signature, []).append(p)

    def best(self) -> Program | None:
        progs = [p for ps in self.clusters.values() for p in ps]
        return min(progs, key=lambda p: (p.excess, p.length)) if progs else None

    def size(self) -> int:
        return sum(len(ps) for ps in self.clusters.values())


class Database:
    def __init__(self, n_islands: int, temperature: float, reset_every: int):
        self.islands = [Island() for _ in range(n_islands)]
        self.temperature = temperature
        self.reset_every = reset_every
        self.resets: list[dict] = []

    def seed(self, programs: list[Program]) -> None:
        for island in self.islands:
            for p in programs:
                island.add(p)

    def add(self, p: Program, island: int) -> None:
        self.islands[island].add(p)

    def best(self) -> Program:
        return min((i.best() for i in self.islands if i.best()), key=lambda p: (p.excess, p.length))

    def cluster_probs(self, island: int) -> tuple[list, np.ndarray]:
        sigs = list(self.islands[island].clusters)
        scores = np.array([self.islands[island].clusters[s][0].score for s in sigs])
        if math.isinf(self.temperature):
            w = np.ones(len(sigs))
        else:
            w = np.exp((scores - scores.max()) / self.temperature)
        return sigs, w / w.sum()

    def sample_parents(self, island: int, rng: np.random.Generator, k: int = 2) -> list[Program]:
        """k parents from distinct clusters, returned worst first."""
        sigs, probs = self.cluster_probs(island)
        k = min(k, len(sigs))
        picked = rng.choice(len(sigs), size=k, replace=False, p=probs)
        parents = []
        for c in picked:
            progs = self.islands[island].clusters[sigs[c]]
            lengths = np.array([p.length for p in progs], dtype=float)
            span = lengths.max() - lengths.min() + 1e-6
            w = np.exp(-(lengths - lengths.min()) / span)
            parents.append(progs[rng.choice(len(progs), p=w / w.sum())])
        return sorted(parents, key=lambda p: -p.excess)

    def maybe_reset(self, step: int, rng: np.random.Generator) -> dict | None:
        """After step `step` (0-based), wipe the worse half if it is time."""
        if (step + 1) % self.reset_every:
            return None
        order = sorted(range(len(self.islands)), key=lambda i: self.islands[i].best().excess)
        keep, wipe = order[: len(order) // 2], order[len(order) // 2:]
        event = {"after_step": step, "wiped": [], "from": []}
        for i in wipe:
            donor = int(rng.choice(keep))
            founder = self.islands[donor].best()
            self.islands[i] = Island()
            self.islands[i].add(founder)
            event["wiped"].append(i)
            event["from"].append(donor)
        self.resets.append(event)
        return event
