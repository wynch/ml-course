"""Section 2: the JEPA objective of pretrain.py, five ways, 120 seconds each.

    uv run python src/collapse.py --variant no-stop-grad    # also stop-grad, ema, vicreg-var, sigreg

A JEPA loss compares two outputs of the same encoder, so it has a trivial
minimum: map every image to one point. Five variants, same encoder, predictor,
masks, optimiser and budget:

  no stop-grad   the target is the online encoder's own output, gradient flows into it
  stop-grad      the same target, detached
  EMA            the target comes from a slow moving average of the encoder (pretrain.py's jepa)
  VICReg var     no stop-grad, plus VICReg's variance hinge on the pooled latents
  SIGReg         no stop-grad, plus SIGReg on the pooled latents (LeJEPA's weighting)

Every 6 seconds it logs the collapse meters of ssl_core.collapse_meters on
2,048 test images; at the end, a class probe from 10,000 training images.
Writes results/collapse_<variant>.json.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import common
import numpy as np
import torch

import ssl_core as S

VARIANTS = {"no-stop-grad": "no stop-grad", "stop-grad": "stop-grad", "ema": "EMA",
            "vicreg-var": "VICReg var", "sigreg": "SIGReg"}
BATCH = 256
LR = 1e-3
VAR_WEIGHT = 1.0      # VICReg variance term: 25 x 0.04, the weights of the source recipe
SIG_LAMBDA = 0.1      # SIGReg: loss = (1 - lambda) * prediction + lambda * SIGReg, as in LeJEPA
EMA = (0.996, 1.0)


def run(variant: str, Xtr, Ytr, Xte, Yte, dev, budget: float) -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    enc, pred = S.Encoder().to(dev), S.Predictor().to(dev)
    tenc = copy.deepcopy(enc).requires_grad_(False) if variant == "EMA" else None
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR, weight_decay=0.05)
    meter_x = Xte[:2048]
    log = [{"step": 0, "seconds": 0.0, "loss": None, "pred_loss": None, **S.collapse_meters(S.embed(enc, meter_x))}]
    t0, step, next_log = time.perf_counter(), 0, 6.0
    while True:
        el = time.perf_counter() - t0
        if el >= budget:
            break
        frac = el / budget
        for g in opt.param_groups:
            g["lr"] = LR * (frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95)))
        x = S.light_view(Xtr[torch.randint(0, len(Xtr), (BATCH,), device=dev)])
        m = S.block_mask(BATCH, rng, dev)
        p = pred(enc.forward_map(S.apply_mask(x, m)), m)
        if variant == "EMA":
            with torch.no_grad():
                tgt = tenc.forward_map(x)
        else:
            tgt = enc.forward_map(x)
            if variant == "stop-grad":
                tgt = tgt.detach()
        lp = S.jepa_loss(p, tgt, m)
        if variant == "VICReg var":
            loss = lp + VAR_WEIGHT * S.vicreg_variance(tgt.mean((2, 3)))
        elif variant == "SIGReg":
            loss = (1 - SIG_LAMBDA) * lp + SIG_LAMBDA * S.sigreg(tgt.mean((2, 3)), gen=gen)
        else:
            loss = lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if tenc is not None:
            S.ema_update(tenc, enc, EMA[0] + (EMA[1] - EMA[0]) * frac)
        step += 1
        if step % 10 == 0:
            common.sync(dev)
            if time.perf_counter() - t0 >= next_log:
                log.append({"step": step, "seconds": round(time.perf_counter() - t0, 1), "loss": round(loss.item(), 6),
                            "pred_loss": round(lp.item(), 6), **S.collapse_meters(S.embed(enc, meter_x))})
                next_log += 6.0
    common.sync(dev)
    Ztr, Zte = S.embed(enc, Xtr[:10000]), S.embed(enc, Xte)
    acc = S.logistic_probe(Ztr, Ytr[:10000].cpu(), Zte, Yte.cpu(), 10)
    out = {"variant": variant, "steps": step, "train_seconds": round(time.perf_counter() - t0, 1),
           "class_acc_10k": round(acc, 4), "final": S.collapse_meters(Zte), "log": log}
    print(variant, {k: out[k] for k in ("steps", "class_acc_10k", "final")}, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--budget", type=float, default=120.0, help="training seconds")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--figures", action="store_true", help="redraw every figure once all results exist")
    args = ap.parse_args()
    guard = common.Guard(f"collapse_{args.variant}")
    dev = common.get_device()
    xtr, ytr, xte, yte = common.load_cifar10()
    Xtr, Ytr = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xte, Yte = torch.from_numpy(xte).to(dev), torch.from_numpy(yte).to(dev)
    guard.tick("load")
    res = run(VARIANTS[args.variant], Xtr, Ytr, Xte, Yte, dev, args.budget)
    guard.tick("train_and_probe")
    res.update({"budget_seconds": args.budget, "batch": BATCH, "lr": LR, "var_weight": VAR_WEIGHT,
                "sigreg_lambda": SIG_LAMBDA, "ema": EMA, "latent_dim": S.D, "encoder_params": S.n_params(S.Encoder()),
                "meter_images": 2048, "probe_train_images": 10000, "device": str(dev)})
    res["guard"] = guard.report()
    if not args.no_write:
        common.write_results(f"collapse_{args.variant}", res)
    if args.figures:
        import figures

        figures.main()


if __name__ == "__main__":
    main()
