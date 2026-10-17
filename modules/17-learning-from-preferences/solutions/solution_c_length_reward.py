"""Solution (c): a reward that sees only length. See exercises/exercise_c_length_reward.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

import data

RESULTS = Path(__file__).resolve().parents[1] / "results"


def length_feature(pairs: list[dict]) -> np.ndarray:
    """log(chosen reply tokens) - log(rejected reply tokens), one per pair."""
    c = np.maximum([p["reply_tokens_chosen"] for p in pairs], 1)       # a reply can be empty
    r = np.maximum([p["reply_tokens_rejected"] for p in pairs], 1)
    return np.log(c) - np.log(r)


def fit_w(d: np.ndarray, steps: int = 2000, lr: float = 0.5) -> float:
    """The w that minimises mean(-log sigmoid(w * d)), by gradient descent from 0."""
    w = 0.0
    for _ in range(steps):
        w -= lr * float(np.mean(-(1 - 1 / (1 + np.exp(-w * d))) * d))
    return w


def main() -> None:
    sp = data.splits(data.load_pairs()[0])
    w = fit_w(length_feature(sp["reward_train"]))
    d_te = length_feature(sp["heldout"])
    acc = float(np.mean(w * d_te > 0) + 0.5 * np.mean(d_te == 0))
    head = json.loads((RESULTS / "reward_head.json").read_text())["heads"]["last_token"]["heldout_accuracy"]
    print(f"w = {w:.3f}; held-out accuracy {acc:.3f} (ties count half); the 960-feature head: {head:.3f}")


if __name__ == "__main__":
    main()
