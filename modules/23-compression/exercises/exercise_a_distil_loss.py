"""Exercise (a): the distillation loss, and what the temperature does to the teacher.

1. ``soften``: the teacher's distribution at temperature T.
2. ``distil_loss``: (1 - alpha) * CE + alpha * T^2 * KL(teacher_T || student_T).

Predict before you run: the teacher gives the next character 'e' a
probability of about one half. At T = 4, is 'e' still on top? Roughly what
probability does it get? And for an untrained student, which is larger, the
alpha = 0 loss or the alpha = 1 loss? Fill in the ``# TODO(you):`` lines, then:

    uv run python ../exercises/exercise_a_distil_loss.py
    uv run pytest tests/test_solutions.py -k distil -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch
import torch.nn.functional as F

CONTEXT = "ROMEO:\nWhat light through yonder window b"


def soften(logits: torch.Tensor, T: float) -> torch.Tensor:
    """softmax(logits / T) over the last dimension."""
    # TODO(you): one line.
    ...


def distil_loss(student_logits, teacher_logits, targets, T: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """(1 - alpha) * CE(student, targets) + alpha * T^2 * KL(soften(teacher) || soften(student)).

    Logits (batch, time, vocab), targets (batch, time). KL is summed over the
    vocabulary and averaged over positions.
    """
    V = student_logits.size(-1)
    s = student_logits.reshape(-1, V)
    t = teacher_logits.reshape(-1, V)
    # TODO(you): ce = cross-entropy of s against the targets;
    # kl = sum over the vocabulary of p_t * (log p_t - log softmax(s / T)), averaged over positions,
    # with p_t = soften(t, T). Return the weighted sum, with T * T on the KL term.
    ...


def main() -> None:
    from tiny_gpt import load_tiny_gpt, new_gpt, splits

    torch.manual_seed(0)
    teacher, tok = load_tiny_gpt("cpu")
    x = torch.tensor([tok.encode(CONTEXT)])
    with torch.no_grad():
        last = teacher(x)[0][0, -1]
    print(f"teacher's next character after {CONTEXT[-12:]!r}:")
    for T in (1.0, 2.0, 4.0):
        p = soften(last, T)
        top = p.topk(5)
        shown = ", ".join(f"{tok.decode([int(i)])!r} {float(v):.2f}" for v, i in zip(top.values, top.indices))
        print(f"  T = {T}: {shown}")

    _, val = splits(tok)
    xb = torch.stack([val[i : i + 128] for i in range(0, 128 * 8, 128)])
    yb = torch.stack([val[i + 1 : i + 129] for i in range(0, 128 * 8, 128)])
    student = new_gpt(88).eval()
    with torch.no_grad():
        s_logits, t_logits = student(xb)[0], teacher(xb)[0]
        for a in (0.0, 0.5, 1.0):
            print(f"untrained student, alpha = {a}: loss {float(distil_loss(s_logits, t_logits, yb, 2.0, a)):.3f}")


if __name__ == "__main__":
    main()
