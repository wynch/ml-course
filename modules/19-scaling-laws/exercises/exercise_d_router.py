"""Exercise (d): a top-k router, the balancing loss, and active parameters.

A mixture-of-experts layer scores E experts for each token, keeps the top k
and mixes their outputs. Write the router (softmax, top-k, renormalise:
Mixtral's rule) and Switch Transformer's balancing loss, E * sum_e f_e P_e,
where f_e is the share of routed slots sent to expert e and P_e the mean
router probability of e. Then count Mixtral 8x7B's active parameters from
its record in the architecture explorer's database.

Predict before you run: Mixtral uses 2 of its 8 experts per token. What share
of its 46.7B parameters does one token use: 25%, more, or less?

    uv run python ../exercises/exercise_d_router.py
    uv run pytest tests/test_solutions.py -k router -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch

RECORD = Path(__file__).resolve().parents[3] / "architectures" / "records" / "mixtral-8x7b.json"


def route(logits: torch.Tensor, k: int):
    """logits (N, E). Softmax over the experts, keep the k largest, renormalise them to sum to 1.
    Returns probs (N, E), weights (N, k), expert ids (N, k)."""
    # TODO(you): softmax, then probs.topk(k, dim=-1), then divide the kept weights by their sum.
    raise NotImplementedError("TODO(you): write route first")


def balance_loss(probs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """E * sum_e f_e P_e. f_e: share of the N*k routed slots sent to e. P_e: mean probability of e."""
    # TODO(you): f from torch.bincount over ids, P from probs.mean(0). Check: uniform routing gives 1.
    raise NotImplementedError("TODO(you): write balance_loss first")


def mixtral_active() -> dict:
    """Mixtral 8x7B's total and active parameters, from its record: every layer has
    E SwiGLU experts of three d x h matrices; a token uses k of them."""
    r = json.loads(RECORD.read_text())
    d, h, L = r["d_model"], r["moe"]["expert_hidden_dim"], r["layers"]
    E, k = r["moe"]["experts"], r["moe"]["routed_per_token"]
    total = r["params"]["total"]
    # TODO(you): the parameters of one expert, then take away the experts a token does not use.
    raise NotImplementedError("TODO(you): write mixtral_active first")


def main() -> None:
    torch.manual_seed(0)
    E, k, N = 8, 2, 4096
    for name, logits in (("random router", torch.randn(N, E) * 0.01),
                         ("router that prefers expert 0", torch.randn(N, E) + torch.tensor([4.0] + [0.0] * 7))):
        probs, w, ids = route(logits, k)
        share = torch.bincount(ids.flatten(), minlength=E).float() / ids.numel()
        print(f"{name:30s} busiest expert {share.max():.2f} of slots (fair 0.125)  balance loss {balance_loss(probs, ids):.3f}")
    m = mixtral_active()
    print(f"Mixtral 8x7B: total {m['total']:,}, active {m['active']:,} (record: {m['record_active']:,})")
    print(f"  uses {m['experts_used_share']:.0%} of its experts, but {m['active_share']:.1%} of its parameters")


if __name__ == "__main__":
    main()
