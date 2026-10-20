"""Solution (b) — ablate the causal mask and expose the leak."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from src import config as C
from src.data import CharTokenizer, get_batch, load_corpus, make_splits
from src.model import GPT, GPTConfig

ITERS = 1500
LR = 1e-3
BATCH = 32
EVAL_BATCHES = 50


def make_noncausal(model: GPT) -> None:
    """Allow attention everywhere: set every mask buffer to all-ones."""
    for block in model.blocks:
        block.attn.mask = torch.ones_like(block.attn.mask)


# --------------------------------------------------------------------------
def make_causal(model: GPT) -> None:
    """Put the lower-triangular mask back, as generation needs it."""
    for block in model.blocks:
        block.attn.mask = torch.tril(torch.ones_like(block.attn.mask))


@torch.no_grad()
def future_leak(model: GPT, tok, device, probe_pos: int = 5) -> float:
    """Max change in the logits AT `probe_pos` when we alter the token at
    `probe_pos + 1` (a strictly-future token). 0.0 means the future cannot leak."""
    model.eval()
    seq = torch.tensor([tok.encode("ROMEO: hello there")], dtype=torch.long, device=device)
    base = model(seq)[0][0, probe_pos]
    alt = seq.clone()
    alt[0, probe_pos + 1] = (alt[0, probe_pos + 1] + 7) % model.cfg.vocab_size
    changed = model(alt)[0][0, probe_pos]
    return (base - changed).abs().max().item()


def train(model, train_data, device, seed: int) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(ITERS):
        x, y = get_batch(train_data, C.BLOCK_SIZE, BATCH, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def val_loss(model, val_data, device) -> float:
    """Mean loss on the same EVAL_BATCHES validation batches for every call."""
    torch.manual_seed(123)
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(val_data, C.BLOCK_SIZE, BATCH, device)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def main() -> dict:
    device = C.get_device()
    text = load_corpus()
    tok = CharTokenizer(text)
    train_data, val_data = make_splits(text, tok)
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

    print(f"\nRetrain: {ITERS} steps each, lr {LR}, batch {BATCH}, same data order")
    train(causal, train_data, device, seed=1)
    train(noncausal, train_data, device, seed=1)
    v_causal = val_loss(causal, val_data, device)
    v_open = val_loss(noncausal, val_data, device)
    make_causal(noncausal)
    v_masked = val_loss(noncausal, val_data, device)
    print(f"  causal model                         val loss = {v_causal:.3f}")
    print(f"  non-causal model, future visible     val loss = {v_open:.3f}")
    print(f"  non-causal model, mask put back      val loss = {v_masked:.3f}")

    print(
        "\nWhy the mask matters:\n"
        "  The non-causal model's prediction at t can read the token at t+1, which\n"
        "  is the label. Learning to exploit that takes a copy circuit; at this size\n"
        "  and budget it may not have formed, so its loss with the future visible\n"
        "  need not be lower than the causal model's. The damage shows at\n"
        "  generation, where the future does not exist: put the mask back and its\n"
        "  loss is far worse. Training and generation must see the same mask."
    )
    return {
        "iters": ITERS, "lr": LR, "batch": BATCH, "eval_batches": EVAL_BATCHES,
        "model": {"n_layer": 2, "n_head": 4, "d_model": 64, "block_size": C.BLOCK_SIZE},
        "device": device,
        "future_leak": {"causal": leak_c, "noncausal": leak_n},
        "val_loss": {"causal": round(v_causal, 4),
                     "noncausal_future_visible": round(v_open, 4),
                     "noncausal_mask_put_back": round(v_masked, 4)},
    }


if __name__ == "__main__":
    result = main()
    out = C.ROOT / "results" / "ex_b_causal_mask.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
