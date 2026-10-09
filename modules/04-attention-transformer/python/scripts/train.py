"""Train the tiny char-level GPT on tiny-shakespeare (MPS).

Records everything the figure scripts need into `models/` (git-ignored):
  - `ckpt_final.pt`          final weights + config + tokenizer vocab
  - `ckpt_step{N}.pt`        checkpoints at 0 / 25 / 50 / 100 % of training
  - `train_log.json`         loss history, sampled text per checkpoint, and a
                             per-checkpoint attention snapshot for one head

Run:  uv run python scripts/train.py
Expected on an Apple M5: ~5-10 min for 5000 iters.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import CharTokenizer, get_batch, load_corpus, make_splits
from src.model import GPT, GPTConfig


def lr_at(it: int) -> float:
    """Linear warmup then cosine decay to 10 % of the peak lr."""
    if it < C.WARMUP_ITERS:
        return C.LEARNING_RATE * (it + 1) / C.WARMUP_ITERS
    frac = (it - C.WARMUP_ITERS) / max(1, C.MAX_ITERS - C.WARMUP_ITERS)
    return C.LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


@torch.no_grad()
def estimate_loss(model, splits, device) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(C.EVAL_ITERS)
        for k in range(C.EVAL_ITERS):
            x, y = get_batch(data, C.BLOCK_SIZE, C.BATCH_SIZE, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(model, tok, device, prompt=C.SAMPLE_PROMPT, n=240) -> str:
    ctx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ctx, n, temperature=0.8, top_k=40)[0].tolist()
    model.train()
    return tok.decode(out)


@torch.no_grad()
def attn_snapshot(model, tok, device, prompt=C.SAMPLE_PROMPT) -> list[list[float]]:
    """Layer-0 head-0 attention matrix for a fixed prompt (for the gif)."""
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    model(ids)
    att = model.blocks[0].attn.last_attn[0, 0].float().cpu().tolist()
    model.train()
    return att


def main() -> None:
    torch.manual_seed(C.SEED)
    device = C.get_device()
    print(f"device = {device}")

    text = load_corpus()
    tok = CharTokenizer(text)
    print(f"corpus = {len(text):,} chars, vocab = {tok.vocab_size}")
    train_data, val_data = make_splits(text, tok)
    splits = {"train": train_data, "val": val_data}

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=C.BLOCK_SIZE,
        n_layer=C.N_LAYER,
        n_head=C.N_HEAD,
        d_model=C.D_MODEL,
        dropout=C.DROPOUT,
    )
    model = GPT(cfg).to(device)
    print(f"parameters = {model.num_params():,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=C.LEARNING_RATE,
        weight_decay=C.WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    checkpoint_iters = {round(f * C.MAX_ITERS) for f in C.CHECKPOINT_FRACTIONS}
    log = {"iters": [], "train": [], "val": [], "checkpoints": []}

    def save_ckpt(tag: str, it: int):
        path = C.MODELS / f"ckpt_{tag}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "chars": [tok.itos[i] for i in range(tok.vocab_size)],
                "iter": it,
            },
            path,
        )
        return path

    t0 = time.time()
    for it in range(C.MAX_ITERS + 1):
        # checkpoint snapshots for the figures
        if it in checkpoint_iters:
            txt = sample_text(model, tok, device)
            att = attn_snapshot(model, tok, device)
            save_ckpt(f"step{it}", it)
            log["checkpoints"].append(
                {"iter": it, "sample": txt, "attn_l0h0": att}
            )
            print(f"\n--- checkpoint @ iter {it} ---\n{txt}\n")

        if it % C.EVAL_INTERVAL == 0 or it == C.MAX_ITERS:
            losses = estimate_loss(model, splits, device)
            log["iters"].append(it)
            log["train"].append(losses["train"])
            log["val"].append(losses["val"])
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {dt:6.1f}s"
            )

        if it == C.MAX_ITERS:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch(train_data, C.BLOCK_SIZE, C.BATCH_SIZE, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    save_ckpt("final", C.MAX_ITERS)
    tok.save(C.MODELS / "tokenizer_chars.json")
    (C.MODELS / "train_log.json").write_text(json.dumps(log))
    print(f"\ntotal training time = {time.time() - t0:.1f}s")
    print(f"final: train {log['train'][-1]:.4f}  val {log['val'][-1]:.4f}")


if __name__ == "__main__":
    main()
