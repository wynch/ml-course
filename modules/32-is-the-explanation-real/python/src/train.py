"""Train the sparse model (and the dense baseline) on TinyStories, in chunks.

Each call trains until about 4.5 minutes have passed, saves a checkpoint to
``artifacts/runs/<name>_ckpt.pt`` and exits; the next call resumes from it.
When the last step is done it evaluates on the held-out stories, writes
``artifacts/<name>.pt`` (float16) and ``results/train_<name>.json``.

Dead neurons: every RESAMPLE steps, a sparse-MLP neuron that never fired in
that window gets a fresh random key (the norm of a typical live key) and a
zero value, and its Adam moments are cleared (after 2610.19907, which
resamples neurons that fire on fewer than 5e-6 of tokens every 1,000 steps).

    uv run python src/train.py sparse      # repeat until it prints "done"
    uv run python src/train.py dense
"""

from __future__ import annotations

import argparse
import math
import time

from common import ARTIFACTS, CHUNK_SECONDS, RUNS, Guard, append_log, get_device, write_results

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import GPT, Config, save_half

STEPS = 8000
BATCH = 64
LR, LR_END, WARMUP = 3e-3, 3e-5, 300
WD = 0.1
RESAMPLE = 1000
SEED = 0
CONFIGS = {"sparse": Config(), "dense": Config(n_mlp=512, k=0)}


def lr_at(step: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_END + 0.5 * (LR - LR_END) * (1 + math.cos(math.pi * t))


def batch(ids: np.ndarray, rng: np.random.Generator, ctx: int, n: int = BATCH):
    s = rng.integers(0, len(ids) - ctx - 1, size=n)
    x = np.stack([ids[i : i + ctx] for i in s])
    y = np.stack([ids[i + 1 : i + ctx + 1] for i in s])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model: GPT, ids: np.ndarray, device, n_windows: int | None = None) -> dict:
    """Mean next-token loss over non-overlapping windows of the held-out stories."""
    ctx = model.c.ctx
    n = (len(ids) - 1) // ctx
    if n_windows:
        n = min(n, n_windows)
    x = torch.from_numpy(ids[: n * ctx].reshape(n, ctx))
    y = torch.from_numpy(ids[1 : n * ctx + 1].reshape(n, ctx))
    tot, cnt, fires = 0.0, 0, None
    for s in range(0, n, 128):
        logits = model(x[s : s + 128].to(device))
        tot += float(F.cross_entropy(logits.flatten(0, 1), y[s : s + 128].to(device).flatten(), reduction="sum"))
        cnt += y[s : s + 128].numel()
        f = torch.stack([(b.mlp.h > 0).flatten(0, 1).float().sum(0) for b in model.blocks])
        fires = f if fires is None else fires + f
    freq = (fires / cnt).cpu().numpy()
    return {"loss": round(tot / cnt, 4), "tokens": cnt, "freq": freq}


def opt_for(model):
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "wpe" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "wpe" not in n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": rest, "weight_decay": 0.0}],
                             lr=LR, betas=(0.9, 0.95))


@torch.no_grad()
def resample(model: GPT, opt, counts: torch.Tensor) -> int:
    n_done = 0
    for li, b in enumerate(model.blocks):
        dead = torch.nonzero(counts[li] == 0).flatten()
        if len(dead) == 0:
            continue
        W_in, W_out = b.mlp.W_in, b.mlp.W_out
        live = counts[li] > 0
        norm = W_in.weight[live].norm(dim=1).median() if live.any() else 0.02 * math.sqrt(W_in.weight.shape[1])
        new = torch.randn(len(dead), W_in.weight.shape[1], device=W_in.weight.device)
        W_in.weight[dead] = new / new.norm(dim=1, keepdim=True) * norm
        W_in.bias[dead] = 0
        W_out.weight[:, dead] = 0
        for p, idx in ((W_in.weight, (dead,)), (W_in.bias, (dead,)), (W_out.weight, (slice(None), dead))):
            st = opt.state.get(p)
            if st:
                st["exp_avg"][idx] = 0
                st["exp_avg_sq"][idx] = 0
        n_done += len(dead)
    return n_done


def run(name: str) -> None:
    guard = Guard(f"train.py {name}")
    dev = get_device()
    c = CONFIGS[name]
    tr, va = data.tokens()
    RUNS.mkdir(parents=True, exist_ok=True)
    ck_path = RUNS / f"{name}_ckpt.pt"
    torch.manual_seed(SEED)
    model = GPT(c).to(dev)
    opt = opt_for(model)
    step, curve, resampled = 0, [], []
    rng = np.random.default_rng(SEED)
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, curve, resampled = ck["step"], ck["curve"], ck["resampled"]
        rng.bit_generator.state = ck["rng"]
        model.to(dev)
    if step >= STEPS:
        print("done (already)")
        return
    model.train()
    counts = torch.zeros(c.layers, c.n_mlp, device=dev)
    t_start = time.perf_counter()
    first = step
    while step < STEPS and time.perf_counter() - t_start < CHUNK_SECONDS:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(tr, rng, c.ctx)
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if c.k:
            with torch.no_grad():
                counts += torch.stack([(b.mlp.h > 0).flatten(0, 1).sum(0) for b in model.blocks]).float()
        step += 1
        if step % 100 == 0:
            curve.append([step, round(loss.item(), 4)])
        if c.k and step % RESAMPLE == 0 and step <= STEPS - 2 * RESAMPLE:
            resampled.append([step, resample(model, opt, counts)])
            counts.zero_()
        if step % RESAMPLE == 0:
            counts.zero_()
    guard.tick("train")
    secs = time.perf_counter() - t_start
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "curve": curve,
                "resampled": resampled, "rng": rng.bit_generator.state}, ck_path)
    g = guard.report()
    append_log(f"train_{name}", {"steps": [first, step], "seconds": round(secs, 1), "peak_rss_gb": g["peak_rss_gb"],
                                 "peak_mps_driver_gb": g["peak_mps_driver_gb"], "last_loss": curve[-1][1] if curve else None})
    print(f"{name}: step {step}/{STEPS}, loss {curve[-1][1] if curve else '-'}, {secs:.0f} s this chunk")
    if step < STEPS:
        return

    # Last chunk: evaluate the float16 copy, the one every later script loads.
    save_half(model, ARTIFACTS / f"{name}.pt", {"steps": STEPS})
    from model import load
    m16 = load(ARTIFACTS / f"{name}.pt", dev)
    ev = evaluate(m16, va, dev)
    freq = ev.pop("freq")
    out = {
        "config": c.dict(),
        "params": sum(p.numel() for p in m16.parameters()) - c.vocab * 0,
        "steps": STEPS, "batch": BATCH, "tokens_seen": STEPS * BATCH * c.ctx,
        "optimizer": f"AdamW betas (0.9, 0.95), lr {LR} warmup {WARMUP} then cosine to {LR_END}, wd {WD}, clip 1.0",
        "val": ev,
        "curve": curve,
        "resampled": resampled,
        "alive_per_layer_1e-5": [int((f >= 1e-5).sum()) for f in freq],
        "dead_per_layer": [int((f == 0).sum()) for f in freq],
        "chunks_log": f"results/train_{name}_log.jsonl",
    }
    out["guard"] = g
    write_results(f"train_{name}", out)
    print("done", ev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(CONFIGS))
    run(ap.parse_args().name)
