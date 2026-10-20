"""Exercise (a): forgetting, and the one number that might predict it.

Predict first: the start model gives the next character "e" probability 0.5,
"a" 0.3 and "o" 0.2. A fine-tuned model gives them 0.8, 0.15 and 0.05. Which is
larger, KL(start || tuned) or KL(tuned || start)? Then run the start model
against itself, and against a copy trained 50 steps on the new text.

    uv run python ../exercises/exercise_a_kl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: E402,F401  (laptop guard before torch)
import torch  # noqa: E402


def forward_kl(logp_ref: torch.Tensor, logp: torch.Tensor) -> float:
    """Mean over positions of KL(ref || model) = sum_c p_ref(c) (log p_ref(c) - log p(c)), in nats.

    Both are log-probabilities of shape (..., vocab).
    """
    # TODO(you): turn logp_ref into probabilities, weight (logp_ref - logp) by them,
    # sum over the last axis, then average over every other axis.
    ...


def forgetting(old_loss_now: float, old_loss_start: float) -> float:
    """How much the loss on the old text rose since the start, in nats per character."""
    # TODO(you): one line.
    ...


def main() -> None:
    p = torch.tensor([0.5, 0.3, 0.2]).log()
    q = torch.tensor([0.8, 0.15, 0.05]).log()
    print(f"KL(start || tuned) = {forward_kl(p, q):.4f}   KL(tuned || start) = {forward_kl(q, p):.4f}")
    if not (common.ARTIFACTS / "start.pt").exists():
        print("artifacts/start.pt is missing: run src/forget.py start first")
        return
    import torch.nn.functional as F

    import data
    import forget

    start = forget.load_start()
    model = forget.load_start()
    x_new, y_new = data.eval_windows()["new"]
    x_old, y_old = data.eval_windows()["old"]
    ref_new, ref_old = forget.logprobs(start, x_new), forget.logprobs(start, x_old)
    print(f"start against itself: KL on the new text {forward_kl(ref_new, forget.logprobs(model, x_new)):.4f}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        model.train()
        x, y = data.batch(data.tensors()["b_train"], 32, gen)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    lp_new, lp_old = forget.logprobs(model, x_new), forget.logprobs(model, x_old)
    old0 = float(F.nll_loss(ref_old.flatten(0, 1), y_old.flatten()))
    old1 = float(F.nll_loss(lp_old.flatten(0, 1), y_old.flatten()))
    print(f"after 50 steps on the new text: KL on the new text {forward_kl(ref_new, lp_new):.3f}, "
          f"on the old text {forward_kl(ref_old, lp_old):.3f}; forgetting {forgetting(old1, old0):.3f} nats/char")


if __name__ == "__main__":
    main()
