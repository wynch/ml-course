"""Exercise (c): a reward that sees only length.

Section 1's reward head reads 960 numbers from the model's residual stream.
Here the reward reads one number: the log of the reply's length in tokens.
Train it with the same Bradley-Terry loss on the same 2,000 pairs, then score
the same 500 held-out pairs.

1. ``length_feature``: for each pair, log(chosen tokens) - log(rejected tokens).
   With r = w * log(tokens), r(chosen) - r(rejected) is w times this.
2. ``fit_w``: gradient descent on mean(-log sigmoid(w * d)) from w = 0.

Needs the cached pairs: run ``uv run python src/data.py`` once (18 MB read,
no model).

Predict before you run: will w come out positive? Will this one-number
reward reach 0.55 held-out accuracy? How far is it from the 960-feature head?

    uv run python ../exercises/exercise_c_length_reward.py
    uv run pytest tests/test_solutions.py -k length -v
"""

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
    # TODO(you): use the keys reply_tokens_chosen and reply_tokens_rejected.
    # A reply can be empty: count lengths as at least 1 before the log.
    ...


def fit_w(d: np.ndarray, steps: int = 2000, lr: float = 0.5) -> float:
    """The w that minimises mean(-log sigmoid(w * d)), by gradient descent from 0."""
    w = 0.0
    for _ in range(steps):
        # TODO(you): the derivative of -log sigmoid(w d) with respect to w is
        # -(1 - sigmoid(w d)) * d; average it over pairs and step.
        ...
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
