"""One training run of module 04's char-GPT on wikitext characters, float or ternary.

Every run sees the same recipe: STEPS steps of BATCH windows of 128
characters, AdamW, warm-up then cosine decay, gradient clipping at 1.0, no
dropout. The batches depend on the seed only, so two arms with the same seed
read the same characters in the same order. Only the shape, the learning rate
and the kind of linear layer change.

    float    plain nn.Linear in float32; rounded after training (section 1)
    ternary  BitLinear (src/bitlinear.py): ternary weights and 8-bit inputs in every forward pass

Losses are mean next-character cross-entropy in nats, on held-out text the
run never trained on.
"""

from __future__ import annotations

import math
import time

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch
import torch.nn.functional as F

from bitlinear import to_bitnet
from tiny_gpt import new_gpt

STEPS = 1000
BATCH = 32
BLOCK = 128
WARMUP = 100
WEIGHT_DECAY = 0.1


def lr_at(step: int, lr: float, steps: int = STEPS) -> float:
    if step < WARMUP:
        return lr * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, steps - WARMUP)
    return lr * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def held_loss(model, ids: np.ndarray, device, batch: int = 128) -> float:
    """Mean next-char CE over every position of `ids`, in windows of BLOCK + 1 that do not overlap."""
    t = torch.from_numpy(ids)
    n = (len(t) - 1) // BLOCK
    x = t[: n * BLOCK].view(n, BLOCK)
    y = t[1 : n * BLOCK + 1].view(n, BLOCK)
    model.eval()
    total = 0.0
    for i in range(0, n, batch):
        logits, _ = model(x[i : i + batch].to(device))
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y[i : i + batch].reshape(-1).to(device), reduction="sum"))
    return total / (n * BLOCK)


def batch_starts(seed: int, n_train: int, steps: int = STEPS) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randint(n_train - BLOCK - 1, (steps, BATCH), generator=g)


def build(kind: str, d: int, n_layer: int, vocab: int):
    model = new_gpt(d, n_layer, vocab_size=vocab, block_size=BLOCK)
    return to_bitnet(model) if kind == "ternary" else model


def train(kind: str, d: int, n_layer: int, lr: float, seed: int, train_ids: np.ndarray, vocab: int, device,
          steps: int = STEPS, ckpt=None, deadline: float | None = None):
    """Train one model; return (model, info), the model on `device` in eval mode.

    With `ckpt` (a path) and `deadline` (a time.perf_counter() value), the run
    stops at the deadline, saves model, optimiser and step to `ckpt`, and
    returns (None, None); the next call with the same arguments resumes there.
    The batches depend only on the seed and the step, so a resumed run trains
    on exactly what an unbroken one would.
    """
    common.seed_all(seed)
    model = build(kind, d, n_layer, vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    first, losses, spent, pieces = 0, [], 0.0, 1
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        first, losses, spent, pieces = state["step"], state["losses"], state["seconds"], state["pieces"] + 1
    data = torch.from_numpy(train_ids)
    starts = batch_starts(seed, len(data), steps)
    offs = torch.arange(BLOCK + 1)
    t0 = time.perf_counter()
    for step in range(first, steps):
        if deadline is not None and time.perf_counter() > deadline:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "losses": losses,
                        "seconds": spent + time.perf_counter() - t0, "pieces": pieces}, ckpt)
            return None, None
        model.train()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, lr, steps)
        chunk = data[starts[step][:, None] + offs].to(device)
        logits, _ = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            losses.append(round(loss.item(), 4))
    model.eval()
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    info = {"kind": kind, "d": d, "n_layer": n_layer, "lr": lr, "seed": seed, "steps": steps,
            "train_loss_every_100": losses, "seconds": round(spent + time.perf_counter() - t0, 1),
            "pieces": pieces, "params": sum(p.numel() for p in model.parameters())}
    return model, info
