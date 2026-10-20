"""Exercise (b): the rule that stands in for auto-interp, and the control that catches it.

Auto-interp asks an LLM to read a unit's top contexts, write an explanation,
then predict from that explanation where the unit fires. Here a rule plays the
LLM: the explanation is the 3 tokens most common among the unit's 50 strongest
activations on the first half of the text. Write:

1. explain(a_half, t_half): those 3 tokens (ties: the token seen first wins).
2. detection(a, t): on the second half, "strongly active" means at least 20%
   of the unit's first-half maximum; predict it iff the current token is in
   the explanation; return balanced accuracy, (TPR + TNR) / 2. Return None if
   the unit fires fewer than 20 times on the first half or is strongly active
   fewer than 5 times on the second.
3. token_r2(a, t, vocab): fit the mean activation per token on the first half
   (unseen tokens get the first half's overall mean), predict the second half
   with it, return 1 - SSE / SST.

Predict first: a unit that fires on exactly one token, always at the same
strength. What are its detection score and token R2? Now a unit that fires
on "the" only when the story is sad: which of the two numbers drops?

    uv run python ../exercises/exercise_b_detection.py
"""

from __future__ import annotations

import numpy as np


def explain(a_half: np.ndarray, t_half: np.ndarray, top: int = 50, n: int = 3) -> np.ndarray:
    # TODO(you)
    raise NotImplementedError


def detection(a: np.ndarray, t: np.ndarray):
    """(score, explanation) or (None, None)."""
    # TODO(you)
    raise NotImplementedError


def token_r2(a: np.ndarray, t: np.ndarray, vocab: int) -> float:
    # TODO(you)
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = rng.integers(0, 50, size=20000)
    one_token = np.where(t == 7, 1.0, 0.0)
    sad = rng.random(20000) < 0.5
    context = np.where((t == 7) & sad, 1.0, 0.0)
    for name, a in (("fires on token 7", one_token), ("token 7, half the time", context)):
        s, e = detection(a, t)
        print(f"{name}: detection {s:.3f} (explanation {e}), token R2 {token_r2(a, t, 50):.3f}")
