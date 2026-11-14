"""Reference solution, exercise (d): a top-k router, the balancing loss, and active parameters."""

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
    probs = logits.softmax(-1)
    w, ids = probs.topk(k, dim=-1)
    return probs, w / w.sum(-1, keepdim=True), ids


def balance_loss(probs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """E * sum_e f_e P_e. f_e: share of the N*k routed slots sent to e. P_e: mean probability of e."""
    E = probs.shape[-1]
    f = torch.bincount(ids.flatten(), minlength=E).float() / ids.numel()
    return E * (f * probs.mean(0)).sum()


def mixtral_active() -> dict:
    """Mixtral 8x7B's total and active parameters, from its record: every layer has
    E SwiGLU experts of three d x h matrices; a token uses k of them."""
    r = json.loads(RECORD.read_text())
    d, h, L = r["d_model"], r["moe"]["expert_hidden_dim"], r["layers"]
    E, k = r["moe"]["experts"], r["moe"]["routed_per_token"]
    expert = 3 * d * h
    total = r["params"]["total"]
    active = total - L * (E - k) * expert
    return {"total": total, "active": active, "record_active": r["params"]["active"],
            "active_share": active / total, "experts_used_share": k / E}


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
