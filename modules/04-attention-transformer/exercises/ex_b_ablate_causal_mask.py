"""Exercise (b) — ablate the causal mask and expose the leak.

The causal mask is the one line that makes a language model a *language* model:
position t may only read positions <= t. Remove it and position t can look at
position t+1 — whose input token is exactly what position t is being asked to
predict. That is the "cheat": the answer leaks in through attention.

Your job: implement `make_noncausal`, which turns off masking in every attention
head. The harness then *proves* the leak two ways:

  1. A **future-leak probe** (deterministic): change a token in the *future* and
     see whether it moves the *current* position's prediction. A causal model is
     provably invariant (change = 0); a non-causal model is not.
  2. A **brief retrain**, so you can watch and explain the consequence.

Run:  uv run python exercises/ex_b_ablate_causal_mask.py   (~1 min on M5)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from src import config as C
from src.data import CharTokenizer, get_batch, load_corpus, make_splits
from src.model import GPT, GPTConfig


def make_noncausal(model: GPT) -> None:
    """Disable the causal mask in every attention block, in place.

    Each `MultiHeadAttention` has a registered buffer `mask` of shape
    (1, 1, block, block): 1 where attention is allowed, 0 where it is forbidden
    (the future). The forward pass does `att.masked_fill(mask == 0, -inf)`. Make
    attention *bidirectional* — allowed everywhere — so nothing is masked out.
    """
    # TODO(you): set every block's attn.mask buffer to all-ones so the
    # masked_fill never fires. hint: torch.ones_like(block.attn.mask)
    raise NotImplementedError("implement make_noncausal")


# --------------------------------------------------------------------------
@torch.no_grad()
def future_leak(model: GPT, tok, device, probe_pos: int = 5) -> float:
    """Max change in the logits AT `probe_pos` when we alter the token at
    `probe_pos + 1` (a strictly-future token). 0.0 ⇒ the future cannot leak."""
    model.eval()
    seq = torch.tensor([tok.encode("ROMEO: hello there")], dtype=torch.long, device=device)
    base = model(seq)[0][0, probe_pos]
    alt = seq.clone()
    alt[0, probe_pos + 1] = (alt[0, probe_pos + 1] + 7) % model.cfg.vocab_size
    changed = model(alt)[0][0, probe_pos]
    return (base - changed).abs().max().item()


def quick_train(model, train_data, device, iters=400):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(iters):
        x, y = get_batch(train_data, C.BLOCK_SIZE, 32, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


def main() -> None:
    torch.manual_seed(0)
    device = C.get_device()
    text = load_corpus()
    tok = CharTokenizer(text)
    train_data, _ = make_splits(text, tok)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=C.BLOCK_SIZE,
                    n_layer=2, n_head=4, d_model=64, dropout=0.0)

    torch.manual_seed(0)
    causal = GPT(cfg).to(device)
    torch.manual_seed(0)
    noncausal = GPT(cfg).to(device)
    make_noncausal(noncausal)

    leak_c = future_leak(causal, tok, device)
    leak_n = future_leak(noncausal, tok, device)
    print("Future-leak probe — does the token at t+1 change the prediction at t?")
    print(f"  causal     model: {leak_c:.6f}   (provably zero — future is masked)")
    print(f"  non-causal model: {leak_n:.6f}   (non-zero — the future leaks in)")
    assert leak_c == 0.0, "a causal model must be invariant to future tokens"
    assert leak_n > 1e-4, "a non-causal model should leak the future"

    print("\nBrief retrain (both models see the same data):")
    print(f"  causal     train loss = {quick_train(causal, train_data, device):.3f}")
    print(f"  non-causal train loss = {quick_train(noncausal, train_data, device):.3f}")

    print(
        "\nWhy the mask matters:\n"
        "  The non-causal model's prediction at t depends on the token at t+1 —\n"
        "  which is the label. Attention hands it a peek at the answer. Trained\n"
        "  long enough it learns to copy rather than to model language, so its loss\n"
        "  stops measuring anything real. Worse, it is useless for GENERATION: at\n"
        "  inference the future does not exist, so a bidirectionally-trained model\n"
        "  is asked to run in a regime it never saw. The causal mask keeps training\n"
        "  and generation consistent — every position predicts the next from the\n"
        "  past alone."
    )


if __name__ == "__main__":
    main()
