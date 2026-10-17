"""Solution (b): the DPO loss by hand. See exercises/exercise_b_dpo_loss.py."""

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
    logp = F.log_softmax(logits[:-1].float(), -1)
    per_token = logp.gather(-1, ids[1:, None])[:, 0]
    return per_token[start - 1:].sum()


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1):
    """Mean over pairs of -log sigmoid(beta * ((pi_w - ref_w) - (pi_l - ref_l)))."""
    z = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(z).mean()


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
