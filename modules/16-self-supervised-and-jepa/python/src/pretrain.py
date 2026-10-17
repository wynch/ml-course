"""Sections 1 and 3: train the one encoder with one objective for a fixed wall-clock
budget, then probe its frozen pooled latents for class, mean colour and rotation.

    uv run python src/pretrain.py --objective simclr       # also mae, jepa, supervised, random
    uv run python src/pretrain.py --objective jepa --budget 30 --no-write   # a quick look

Every objective trains the same Encoder (ssl_core.py) for the same number of
seconds; `random` trains for zero and gives the floor. Writes
results/pretrain_<objective>.json.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import common
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl_core as S

OBJECTIVES = ("random", "supervised", "simclr", "mae", "jepa")
BATCH = {"supervised": 256, "simclr": 512, "mae": 256, "jepa": 256}
MASK_RATIO = 0.6            # masked reconstruction: share of the 64 cells hidden
TAU = 0.2                   # SimCLR temperature
EMA = (0.996, 1.0)          # JEPA target momentum, start and end (I-JEPA's values)
SAMPLE = 7                  # test image shown in the views figure


def build(objective: str, dev):
    torch.manual_seed(0)
    enc = S.Encoder().to(dev)
    heads = nn.ModuleDict()
    if objective == "supervised":
        heads["cls"] = nn.Linear(S.D, 10)
    elif objective == "simclr":
        heads["proj"] = nn.Sequential(nn.Linear(S.D, 256), nn.ReLU(), nn.Linear(256, 128))
    elif objective == "mae":
        heads["dec"] = S.PixelDecoder()
    elif objective == "jepa":
        heads["pred"] = S.Predictor()
    return enc, heads.to(dev)


def to_uint8_img(x: torch.Tensor) -> list:
    """A normalised (3, 32, 32) image back to nested uint8 lists (32, 32, 3)."""
    m = torch.tensor(S.MEAN, device=x.device).view(3, 1, 1)
    s = torch.tensor(S.STD, device=x.device).view(3, 1, 1)
    return (((x * s + m).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()).tolist()


def sample_views(objective, Xte, dev, rng) -> dict | None:
    """What the objective sees for one test image, for the views figure."""
    g = torch.Generator(device=dev).manual_seed(1)
    xb = Xte[SAMPLE:SAMPLE + 1]
    orig = S.normalize(S.to_float(xb))[0]
    if objective == "simclr":
        return {"original": to_uint8_img(orig), "view1": to_uint8_img(S.simclr_view(xb, g)[0]),
                "view2": to_uint8_img(S.simclr_view(xb, g)[0])}
    if objective == "mae":
        m = S.random_patch_mask(1, MASK_RATIO, dev, g)
        return {"original": to_uint8_img(orig), "masked": to_uint8_img(S.apply_mask(orig[None], m)[0]),
                "mask": m[0].int().tolist()}
    if objective == "jepa":
        m = S.block_mask(1, np.random.default_rng(1), dev)
        return {"original": to_uint8_img(orig), "context": to_uint8_img(S.apply_mask(orig[None], m)[0]),
                "target_mask": m[0].int().tolist()}
    return None


def step_loss(objective, enc, heads, tenc, xb, ytr_b, dev, rng):
    if objective == "supervised":
        return F.cross_entropy(heads["cls"](enc(S.light_view(xb))), ytr_b)
    if objective == "simclr":
        z1 = heads["proj"](enc(S.simclr_view(xb)))
        z2 = heads["proj"](enc(S.simclr_view(xb)))
        return S.nt_xent(z1, z2, TAU)
    x = S.light_view(xb)
    if objective == "mae":
        m = S.random_patch_mask(len(x), MASK_RATIO, dev)
        rec = heads["dec"](enc.forward_map(S.apply_mask(x, m)))
        up = m.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1).float()
        return (((rec - x) ** 2) * up).sum() / (up.sum() * 3)
    if objective == "jepa":
        m = S.block_mask(len(x), rng, dev)
        pred = heads["pred"](enc.forward_map(S.apply_mask(x, m)), m)
        with torch.no_grad():
            tgt = tenc.forward_map(x)
        return S.jepa_loss(pred, tgt, m)
    raise ValueError(objective)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=OBJECTIVES, required=True)
    ap.add_argument("--budget", type=float, default=250.0, help="training seconds")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--figures", action="store_true", help="redraw every figure once all results exist")
    args = ap.parse_args()

    guard = common.Guard(f"pretrain_{args.objective}")
    dev = common.get_device()
    xtr, ytr, xte, yte = common.load_cifar10()
    Xtr, Ytr = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xte, Yte = torch.from_numpy(xte).to(dev), torch.from_numpy(yte).to(dev)
    guard.tick("load")

    enc, heads = build(args.objective, dev)
    tenc = copy.deepcopy(enc).requires_grad_(False) if args.objective == "jepa" else None
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    budget = 0.0 if args.objective == "random" else args.budget
    bs = BATCH.get(args.objective, 256)
    lr = 1e-3 * bs / 256
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=lr, weight_decay=0.05)
    meter_x = Xte[:2048]
    log = [{"step": 0, "seconds": 0.0, "loss": None, **S.collapse_meters(S.embed(enc, meter_x))}]

    t0, step, next_log = time.perf_counter(), 0, 10.0
    while budget > 0:
        el = time.perf_counter() - t0
        if el >= budget:
            break
        frac = el / budget
        for g in opt.param_groups:
            g["lr"] = lr * (frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95)))
        idx = torch.randint(0, len(Xtr), (bs,), device=dev)
        loss = step_loss(args.objective, enc, heads, tenc, Xtr[idx], Ytr[idx], dev, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if tenc is not None:
            S.ema_update(tenc, enc, EMA[0] + (EMA[1] - EMA[0]) * frac)
        step += 1
        if step % 20 == 0:
            common.sync(dev)
            if time.perf_counter() - t0 >= next_log:
                lv = loss.item()
                if not math.isfinite(lv):
                    raise SystemExit(f"non-finite loss at step {step}")
                log.append({"step": step, "seconds": round(time.perf_counter() - t0, 1), "loss": round(lv, 5),
                            **S.collapse_meters(S.embed(enc, meter_x))})
                print(log[-1], flush=True)
                next_log += 10.0
                guard.tick()
    common.sync(dev)
    train_seconds = round(time.perf_counter() - t0, 1) if budget > 0 else 0.0
    guard.tick("train")

    probes = S.all_probes(enc, Xtr, Ytr, Xte, Yte)
    guard.tick("probe")
    print(probes)
    res = {
        "objective": args.objective,
        "encoder_params": S.n_params(enc),
        "head_params": S.n_params(heads),
        "latent_dim": S.D,
        "budget_seconds": budget,
        "train_seconds": train_seconds,
        "steps": step,
        "batch": bs if budget > 0 else 0,
        "images_seen": step * bs,
        "lr": lr if budget > 0 else 0,
        "probes": probes,
        "log": log,
        "sample": sample_views(args.objective, Xte, dev, rng),
        "config": {"mask_ratio": MASK_RATIO, "tau": TAU, "ema": EMA, "weight_decay": 0.05,
                   "schedule": "linear warmup over the first 5% of the budget, then cosine to 0, keyed on wall time"},
        "data": {"id": common.CIFAR_ID, "revision": common.CIFAR_REVISION, "train": len(xtr), "test": len(xte)},
        "device": str(dev),
    }
    res["guard"] = guard.report()
    if not args.no_write:
        common.write_results(f"pretrain_{args.objective}", res)
    if args.figures:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
