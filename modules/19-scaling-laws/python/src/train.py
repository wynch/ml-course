"""Section 1: train one size of char-GPT on 4M characters of wikitext; log held-out loss.

    uv run python src/train.py --d 32        # one size per call; six calls in all
    uv run python src/train.py --figures     # redraw the loss curves from results/

Each call trains module 04's GPT (4 layers, 4 heads, width d, dropout 0) for
977 steps of 32 x 128 characters, one pass over the training part with no
window seen twice. The learning rate warms up over 100 steps to 3e-4 and then
stays there; with no decay, the loss after any number of steps is that of a
run stopped there. The held-out loss is measured at twelve points. Sizes 32
and 48 also run seed 1, the noise band. Writes results/train_d<d>.json.
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch

from char_gpt import GPT, GPTConfig
from common import (BATCH, BLOCK, EVAL_STEPS, LR, N_HEAD, N_LAYER, SIZES, STEPS, TWO_SEEDS, WARMUP,
                    WEIGHT_DECAY, Guard, get_device, tokens_at, write_results)
from corpus import splits


def build(vocab: int, d: int, seed: int) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, block_size=BLOCK, n_layer=N_LAYER, n_head=N_HEAD, d_model=d, dropout=0.0)
    return GPT(cfg)


def held_batches(held: np.ndarray):
    """390 non-overlapping windows of 128 inputs and their next characters."""
    n = (len(held) - 1) // BLOCK
    x = torch.from_numpy(held[: n * BLOCK].reshape(n, BLOCK))
    y = torch.from_numpy(held[1: n * BLOCK + 1].reshape(n, BLOCK))
    return x, y


@torch.no_grad()
def held_loss(model: GPT, hx: torch.Tensor, hy: torch.Tensor, chunk: int = 130) -> float:
    """Mean cross-entropy in nats per character over every held-out window."""
    was = model.training
    model.eval()
    tot = 0.0
    for i in range(0, len(hx), chunk):
        _, loss = model(hx[i:i + chunk], hy[i:i + chunk])
        tot += loss.item() * len(hx[i:i + chunk])
    model.train(was)
    return tot / len(hx)


def train_one(d: int, seed: int, train: np.ndarray, held: np.ndarray, vocab: int, dev, guard: Guard) -> tuple[dict, GPT]:
    model = build(vocab, d, seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)

    w = BLOCK + 1
    n_win = len(train) // w
    order = np.random.default_rng(seed).permutation(n_win)   # the seed sets init and data order
    wins = torch.from_numpy(train[: n_win * w].reshape(n_win, w))
    hx, hy = (t.to(dev) for t in held_batches(held))

    curve, train_loss = [], []
    model.train()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, step / WARMUP)
        b = wins[order[(step - 1) * BATCH: step * BATCH]].to(dev)
        _, loss = model(b[:, :-1], b[:, 1:].contiguous())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0:
            train_loss.append([step, round(loss.item(), 5)])
        if step in EVAL_STEPS:
            curve.append({"step": step, "tokens": tokens_at(step), "held_loss": round(held_loss(model, hx, hy), 6)})
            guard.tick()
    print(f"  d {d} seed {seed}: N {model.num_params_non_embedding():,}, held-out loss {curve[-1]['held_loss']:.4f}")
    run = {"seed": seed, "n_params_total": model.num_params(), "n_non_embedding": model.num_params_non_embedding(),
           "curve": curve, "train_loss": train_loss}
    return run, model


@torch.no_grad()
def mps_vs_cpu(model: GPT, held: np.ndarray, dev) -> dict:
    """One forward pass on 8 held-out windows, on MPS and on the CPU."""
    hx, hy = held_batches(held)
    x, y = hx[:8], hy[:8]
    model.eval()
    a, la = model(x.to(dev), y.to(dev))
    cpu = model.to("cpu")
    b, lb = cpu(x, y)
    model.to(dev)
    a = a.cpu()
    return {"device": dev.type, "windows": 8, "max_abs_diff_logits": float((a - b).abs().max()),
            "logit_range": [float(b.min()), float(b.max())], "loss_mps": float(la), "loss_cpu": float(lb),
            "same_argmax_share": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


@torch.no_grad()
def sample(model: GPT, vocab, prompt: str = "The ", n: int = 160, seed: int = 0) -> str:
    """Sampled text at temperature 1, CPU generator for repeatability."""
    model.eval().to("cpu")
    torch.manual_seed(seed)
    idx = torch.from_numpy(vocab.encode(prompt)).view(1, -1)
    out = model.generate(idx, n, temperature=1.0)
    return vocab.decode(out[0].tolist())


def run(d: int) -> None:
    guard = Guard(f"train.py --d {d}")
    dev = get_device()
    train, held, vocab = splits()
    guard.tick("load data")
    runs, model = [], None
    for seed in ((0, 1) if d in TWO_SEEDS else (0,)):
        r, m = train_one(d, seed, train, held, vocab.size, dev, guard)
        runs.append(r)
        if seed == 0:
            model = m
        guard.tick(f"train seed {seed}")
    check = mps_vs_cpu(model, held, dev) if dev.type == "mps" else None
    text = sample(model, vocab)
    guard.tick("check and sample")
    write_results(f"train_d{d}", {
        "d_model": d, "n_layer": N_LAYER, "n_head": N_HEAD, "block": BLOCK, "batch": BATCH, "steps": STEPS,
        "warmup": WARMUP, "lr": LR, "weight_decay": WEIGHT_DECAY, "vocab": vocab.size,
        "train_chars": int(len(train)), "held_chars": int(len(held)),
        "held_chars_predicted": int(held_batches(held)[0].numel()),
        "device": dev.type, "runs": runs, "mps_vs_cpu": check, "sample_seed0": text,
        "guard": guard.report(),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, choices=SIZES)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.d is not None:
        run(args.d)
    if args.figures:
        import figures
        figures.fig_curves()


if __name__ == "__main__":
    main()
