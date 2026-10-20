"""Solution (a): ternary weights by absmean, the straight-through gradient, and ternary after the fact.

    uv run python ../solutions/solution_a_bitlinear.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def ternary_codes(W: torch.Tensor) -> torch.Tensor:
    """g = mean|W|; code = clamp(round(W / g), -1, 1)."""
    g = W.abs().mean().clamp_min(1e-5)
    return (W / g).round().clamp(-1, 1)


def ternary(W: torch.Tensor) -> torch.Tensor:
    """code * mean|W|."""
    return ternary_codes(W) * W.abs().mean().clamp_min(1e-5)


def ste_ternary(W: torch.Tensor) -> torch.Tensor:
    """Forward: ternary(W). Backward: the gradient of W, as if the rounding were not there."""
    return W + (ternary(W) - W).detach()


def main() -> None:
    from tiny_gpt import load_tiny_gpt, val_text

    torch.manual_seed(0)
    W = torch.randn(64, 64, requires_grad=True)
    c = ternary_codes(W.detach())
    print("share of codes -1, 0, 1:", [round(float((c == v).float().mean()), 3) for v in (-1, 0, 1)])
    ste_ternary(W).sum().backward()
    print("gradient through the STE is all ones:", bool((W.grad == 1).all()))
    W2 = W.detach().clone().requires_grad_(True)
    ternary(W2).sum().backward()
    # Without the STE, round() passes no gradient; only the scale g does, the same size for every weight.
    print("gradient without the STE: largest |entry|", f"{float(W2.grad.abs().max()):.4f}",
          "| distinct sizes:", len(torch.unique(W2.grad.abs())))

    model, tok = load_tiny_gpt("cpu")
    ids = torch.tensor(tok.encode(val_text()[:20_001]))
    x, y = ids[:-1].view(-1, 125), ids[1:].view(-1, 125)          # 160 windows of 125 characters

    @torch.no_grad()
    def loss():
        logits, _ = model(x)
        return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))

    before = loss()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("blocks") and name.endswith("weight") and p.dim() == 2:
                p.copy_(ternary(p))
    after = loss()
    print(f"module 04's char-GPT on 20,000 validation characters: float {before:.3f}, "
          f"rounded to ternary after training {after:.3f} nats")


if __name__ == "__main__":
    main()
