"""Section 2: SGD, momentum and Adam on module 02's MLP, four learning rates
each, with a constant rate and with linear warmup then cosine decay.

FashionMNIST, module 02's split (20,000 training images, the full 10,000 test
images), batch 128, 10 epochs = 1,570 steps. Every run starts from the same
weights and sees the batches in the same order (seed 0). Warmup is the first
10% of steps. The best rate of each optimiser and schedule is then re-run
with seeds 1 and 2, to show how much of a gap is seed noise.

Runs on the CPU: the net is small enough that the GPU does not pay.

    uv run python src/optim_mlp.py [--figures]
"""

from __future__ import annotations

import argparse
import math

from common import MLP, Guard, evaluate, load_fashion, write_results

import torch

LRS = [3e-4, 3e-3, 3e-2, 3e-1]
OPTIMISERS = ("sgd", "momentum", "adam")
SCHEDULES = ("constant", "warmup_cosine")
EPOCHS = 10
BATCH = 128
WARMUP_FRAC = 0.1
THREADS = 4


def make_optimiser(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Multiplier on the base rate at ``step`` (0-based): 0 -> 1 linearly over
    ``warmup`` steps, then half a cosine from 1 down to 0 at ``total``.
    Same formula as transformers.get_cosine_schedule_with_warmup."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_one(data, opt_name: str, lr: float, schedule: str, seed: int = 0, epochs: int = EPOCHS):
    x_tr, y_tr, x_te, y_te = data
    torch.manual_seed(seed)
    model = MLP()
    opt = make_optimiser(opt_name, model.parameters(), lr)
    steps_per_epoch = math.ceil(len(x_tr) / BATCH)
    total = epochs * steps_per_epoch
    warmup = round(WARMUP_FRAC * total)
    g = torch.Generator().manual_seed(seed)
    step, curve, diverged = 0, [], False
    for epoch in range(epochs):
        perm = torch.randperm(len(x_tr), generator=g)
        run_loss = 0.0
        for i in range(0, len(x_tr), BATCH):
            idx = perm[i:i + BATCH]
            mult = warmup_cosine(step, warmup, total) if schedule == "warmup_cosine" else 1.0
            for group in opt.param_groups:
                group["lr"] = lr * mult
            loss = torch.nn.functional.cross_entropy(model(x_tr[idx]), y_tr[idx])
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            step += 1
        if diverged:
            break
        te_loss, te_acc = evaluate(model, x_te, y_te)
        curve.append({"epoch": epoch + 1, "train_loss": round(run_loss / len(x_tr), 4),
                      "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 4)})
    if diverged:
        return {"diverged": True, "curve": curve, "test_acc": None, "train_loss": None, "steps": step}
    tr_loss, tr_acc = evaluate(model, x_tr, y_tr)
    return {"diverged": False, "curve": curve, "test_acc": curve[-1]["test_acc"], "test_loss": curve[-1]["test_loss"],
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4), "steps": step}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    guard = Guard("optim_mlp")
    data = load_fashion()
    guard.tick("load")

    runs = []
    for opt_name in OPTIMISERS:
        for schedule in SCHEDULES:
            for lr in LRS:
                r = train_one(data, opt_name, lr, schedule)
                runs.append({"optimiser": opt_name, "schedule": schedule, "lr": lr, "seed": 0, **r})
                print(f"{opt_name:8s} {schedule:13s} lr {lr:<7g} test acc {r['test_acc']}  train loss {r['train_loss']}")
    guard.tick("grid")

    best = {}
    for opt_name in OPTIMISERS:
        for schedule in SCHEDULES:
            cands = [r for r in runs if r["optimiser"] == opt_name and r["schedule"] == schedule and not r["diverged"]]
            b = max(cands, key=lambda r: r["test_acc"])
            accs = [b["test_acc"]]
            for seed in (1, 2):
                r = train_one(data, opt_name, b["lr"], schedule, seed=seed)
                runs.append({"optimiser": opt_name, "schedule": schedule, "lr": b["lr"], "seed": seed, **r})
                accs.append(r["test_acc"])
            best[f"{opt_name}/{schedule}"] = {"lr": b["lr"], "test_acc_seeds": accs,
                                              "mean": round(sum(accs) / 3, 4), "spread": round(max(accs) - min(accs), 4)}
            print(f"best {opt_name}/{schedule}: lr {b['lr']}, seeds {accs}")
    guard.tick("seeds")

    out = {"lrs": LRS, "epochs": EPOCHS, "batch": BATCH, "warmup_frac": WARMUP_FRAC,
           "steps": runs[0]["steps"], "n_train": len(data[0]), "n_test": len(data[2]),
           "device": "cpu", "threads": THREADS, "runs": runs, "best": best}
    out["guard"] = guard.report()
    write_results("optim_mlp", out)
    if args.figures:
        import figures
        figures.fig_optim_mlp()


if __name__ == "__main__":
    main()
