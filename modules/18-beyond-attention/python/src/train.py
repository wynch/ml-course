"""One training recipe for every model in this module.

AdamW (betas 0.9, 0.99, weight decay 0.1, as in module 04), peak learning rate
1e-3, 100 warmup steps, cosine decay to 10% of the peak, gradient clipping at
1.0, batches of 64 windows of 128 characters. No dropout. The recipe is not
tuned per model: a fair start, not the best each model can do.

The run stops after ``steps`` steps or after ``seconds`` of training time,
whichever is given. With ``seconds``, the cosine follows elapsed time instead
of steps, so every model gets a full schedule inside its budget. Evaluation
time is not counted.
"""

from __future__ import annotations

import math
import time

import torch

from common import CharData, sync

PEAK_LR = 1e-3
WARMUP = 100


def lr_at(step: int, progress: float) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    return PEAK_LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.no_grad()
def evaluate(model, windows, device) -> float:
    """Mean next-character loss (nats) over a fixed list of (x, y) batches."""
    model.eval()
    total = 0.0
    for x, y in windows:
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / len(windows)


def train(model, data: CharData, device, *, steps: int | None = None, seconds: float | None = None,
          seed: int = 0, eval_every: float = 0.1, val_quick=None, val_full=None) -> dict:
    """Train in place. Returns the loss log and the final validation loss.

    eval_every: fraction of the run between quick validation checks.
    val_quick / val_full: fixed lists of validation batches (CharData.val_windows).
    """
    assert (steps is None) != (seconds is None)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    val_quick = val_quick or data.val_windows(10)
    val_full = val_full or data.val_windows(100)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.99), weight_decay=0.1)

    log = {"step": [], "seconds": [], "train": [], "val_step": [], "val_seconds": [], "val": []}
    trained = 0.0                       # seconds spent training, evaluation excluded
    next_eval, step, window = 0.0, 0, []
    while True:
        progress = step / steps if steps else trained / seconds
        if progress >= next_eval or progress >= 1.0:
            log["val_step"].append(step)
            log["val_seconds"].append(round(trained, 2))
            log["val"].append(round(evaluate(model, val_quick, device), 4))
            next_eval += eval_every
        if progress >= 1.0:
            break
        sync(device)
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, progress)
        x, y = data.batch("train", gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        window.append(loss.item())      # .item() waits for the GPU, so the clock below is honest
        trained += time.perf_counter() - t0
        step += 1
        if len(window) == 25:
            log["step"].append(step)
            log["seconds"].append(round(trained, 2))
            log["train"].append(round(sum(window) / len(window), 4))
            window = []
    val = evaluate(model, val_full, device)
    model.eval()
    return {
        "steps": step,
        "train_seconds": round(trained, 1),
        "ms_per_step": round(1000 * trained / max(step, 1), 1),
        "train_tokens_per_s": round(step * x.numel() / trained),
        "val_loss": round(val, 4),
        "log": log,
    }


@torch.no_grad()
def sample(model, data: CharData, device, prompt: str = "ROMEO:", n: int = 200, seed: int = 0) -> str:
    """Generate with model.step, temperature 0.8 and top-k 40, as module 04 does."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    stepper = model
    state = stepper.init_state(1, device)
    ids = data.encode(prompt)
    for i in ids:
        logits, state = stepper.step(torch.tensor([i], device=device), state)
    out = list(ids)
    for _ in range(n):
        logits = logits[0].float().cpu() / 0.8
        v, _ = torch.topk(logits, 40)
        logits[logits < v[-1]] = -float("inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        out.append(nxt)
        logits, state = stepper.step(torch.tensor([nxt], device=device), state)
    return data.decode(out)
