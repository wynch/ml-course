"""Exercise (b): the DPO loss by hand.

1. ``reply_logp``: from a model's logits over one sequence, the sum of the
   log-probabilities of the reply's tokens. Position t's logits predict token
   t + 1, so the reply's first token (index ``start``) is predicted by
   position start - 1.
2. ``dpo_loss``: from the four sums (policy and reference, chosen and
   rejected), the mean of -log sigmoid(beta * (chosen log-ratio - rejected
   log-ratio)).

The script then runs your loss on the eight held-out pairs that
src/dpo_by_hand.py stored, and prints TRL's number beside it.

Predict before you run: when the policy equals the reference, what is the
loss of every pair, whatever beta is? What happens to the loss of this batch
as beta grows from 0.01 to 0.5?

    uv run python ../exercises/exercise_b_dpo_loss.py
    uv run pytest tests/test_solutions.py -k dpo -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F

RESULTS = Path(__file__).resolve().parents[1] / "results"


def reply_logp(logits: torch.Tensor, ids: torch.Tensor, start: int) -> torch.Tensor:
    """Sum of log p(ids[t] | ids[:t]) for t = start .. len(ids)-1. logits: (T, V), ids: (T,)."""
    # TODO(you): log_softmax over the vocabulary, drop the last position,
    # gather the log-prob of each next token, sum from the right place.
    ...


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1):
    """Mean over pairs of -log sigmoid(beta * ((pi_w - ref_w) - (pi_l - ref_l)))."""
    # TODO(you): two lines; F.logsigmoid is stable for large negative inputs.
    ...


def main() -> None:
    r = json.loads((RESULTS / "dpo_by_hand.json").read_text())
    t = lambda k: torch.tensor([p[k] for p in r["per_pair"]])            # noqa: E731
    pi_w, pi_l, ref_w, ref_l = t("pi_chosen"), t("pi_rejected"), t("ref_chosen"), t("ref_rejected")
    print(f"loss on the stored batch: {float(dpo_loss(pi_w, pi_l, ref_w, ref_l)):.6f} "
          f"(TRL: {r['trl']['loss']:.6f})")
    print(f"policy = reference: {float(dpo_loss(ref_w, ref_l, ref_w, ref_l)):.6f}, log 2 = {math.log(2):.6f}")
    for beta in (0.01, 0.1, 0.5):
        print(f"  beta {beta}: loss {float(dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)):.4f}")


if __name__ == "__main__":
    main()
