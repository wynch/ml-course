"""Exercise (c): the policy-gradient loss, and how a reply's length weighs in it.

Each reply i has an advantage A_i and per-character log-probabilities. The
loss pushes every character of reply i by A_i. GRPO first averages over the
reply's own length, so each character of a long reply gets a smaller share;
Dr. GRPO divides every reply by the same constant (the longest allowed reply).

Predict first: two wrong replies with the same advantage, -1, one 4
characters long, one 28. Under GRPO, how much smaller is the push on each
character of the long one? Under Dr. GRPO?

    uv run python ../exercises/exercise_c_loss.py
"""

from __future__ import annotations

import torch

MAX_REPLY = 32


def pg_loss(lp, lp_old, adv, mask, length_norm: bool, eps: float = 0.2):
    """PPO-clip loss over replies. lp, lp_old, mask: (N, T); adv: (N,)."""
    ratio = torch.exp(lp - lp_old)
    a = adv[:, None]
    per_tok = -torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a) * mask
    # TODO(you): length_norm=True: sum each reply's characters and divide by its own length
    # (mask.sum(1)); False: divide every reply's sum by MAX_REPLY. Then average over replies.
    ...


def main() -> None:
    T = MAX_REPLY
    mask = torch.zeros(2, T)
    mask[0, :4] = 1          # a short reply
    mask[1, :28] = 1         # a long reply
    adv = torch.tensor([-1.0, -1.0])
    for norm in (True, False):
        lp = torch.full((2, T), -0.5, requires_grad=True)
        pg_loss(lp, lp.detach(), adv, mask, norm).backward()
        g = lp.grad
        print("GRPO (per-reply mean)" if norm else "Dr. GRPO (constant)  ",
              "push on each character (negative = down): short", round(-float(g[0, 0]), 4), " long", round(-float(g[1, 0]), 4))


if __name__ == "__main__":
    main()
