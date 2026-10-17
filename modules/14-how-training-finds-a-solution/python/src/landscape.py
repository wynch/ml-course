"""Section 3: what the loss looks like around a trained net, and how sharp it is.

Two copies of module 02's MLP, same initial weights, SGD with momentum 0.9 at
rate 0.05: one with batch 32, one with batch 2,048. Each trains until its loss
on the whole training set first drops to 0.25 or below (checked after every
epoch), so both stop at about the same training loss and differ in how they
got there.

Then, for each net:

* a 1-D slice: loss at theta + a d, for a in [-1, 1], with d a random
  direction rescaled filter by filter (each row of each weight matrix of d gets
  the norm of the same row of theta; bias entries of d are zero), as in Li et
  al., "Visualizing the Loss Landscape of Neural Nets" (2018);
* a 2-D slice: theta + a d1 + b d2, two such directions;
* the top eigenvalue of the Hessian of the training loss, by power iteration
  on Hessian-vector products.

Losses are measured on fixed 5,000-image subsets (training and test), the same
for both nets. Runs on the CPU.

    uv run python src/landscape.py [--figures]
"""

from __future__ import annotations

import argparse

from common import MLP, Guard, evaluate, load_fashion, write_results

import torch

LR = 0.05
MOMENTUM = 0.9
BATCHES = {"small": 32, "large": 2048}
TARGET_TRAIN_LOSS = 0.25
MAX_EPOCHS = 400
SUBSET = 5000
LINE = [round(-1 + 2 * i / 50, 2) for i in range(51)]
GRID = [round(-1 + 2 * i / 30, 4) for i in range(31)]
POWER_ITERS = 100
THREADS = 4


def train_to_target(data, batch: int, seed: int = 0):
    x_tr, y_tr, x_te, y_te = data
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    g = torch.Generator().manual_seed(seed)
    steps, history = 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(len(x_tr), generator=g)
        for i in range(0, len(x_tr), batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
        tr_loss, tr_acc = evaluate(model, x_tr, y_tr)
        history.append({"epoch": epoch, "steps": steps, "train_loss": round(tr_loss, 4)})
        if tr_loss <= TARGET_TRAIN_LOSS:
            break
    te_loss, te_acc = evaluate(model, x_te, y_te)
    return model, {"batch": batch, "epochs": epoch, "steps": steps, "reached_target": tr_loss <= TARGET_TRAIN_LOSS,
                   "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                   "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 4), "history": history}


def filter_normalised_direction(model, seed: int):
    """Random direction with each weight row scaled to the norm of the matching row of the model."""
    g = torch.Generator().manual_seed(seed)
    d = []
    for p in model.parameters():
        r = torch.randn(p.shape, generator=g)
        if p.dim() == 1:
            r.zero_()                                            # biases stay put
        else:
            r = r * (p.detach().norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True))
        d.append(r)
    return d


@torch.no_grad()
def loss_at(model, theta, dirs, coefs, x, y) -> float:
    for p, t, *ds in zip(model.parameters(), theta, *dirs):
        p.copy_(t + sum(c * d for c, d in zip(coefs, ds)))
    loss = torch.nn.functional.cross_entropy(model(x), y).item()
    for p, t in zip(model.parameters(), theta):
        p.copy_(t)
    return loss


def hvp(model, x, y, v):
    """Hessian of the mean cross-entropy on (x, y), times the vector v (a list shaped like the parameters)."""
    params = list(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * u).sum() for g, u in zip(grads, v))
    return torch.autograd.grad(dot, params)


def top_eigenvalue(model, x, y, iters: int = POWER_ITERS, seed: int = 0):
    """Largest Hessian eigenvalue by power iteration. Returns the estimate after each iteration."""
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g) for p in model.parameters()]
    norm = torch.sqrt(sum((u * u).sum() for u in v))
    v = [u / norm for u in v]
    trace = []
    for _ in range(iters):
        hv = hvp(model, x, y, v)
        lam = sum((h * u).sum() for h, u in zip(hv, v)).item()      # Rayleigh quotient, |v| = 1
        trace.append(lam)
        norm = torch.sqrt(sum((h * h).sum() for h in hv))
        v = [h / norm for h in hv]
    return trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    guard = Guard("landscape")
    data = load_fashion()
    x_tr, y_tr, x_te, y_te = data
    xs, ys = x_tr[:SUBSET], y_tr[:SUBSET]
    xt, yt = x_te[:SUBSET], y_te[:SUBSET]
    guard.tick("load")

    torch.manual_seed(0)
    init = MLP()
    init_trace = top_eigenvalue(init, xs, ys)
    out = {"lr": LR, "momentum": MOMENTUM, "target_train_loss": TARGET_TRAIN_LOSS, "subset": SUBSET,
           "line": LINE, "grid": GRID, "power_iters": POWER_ITERS,
           "stability_limit": round(2 * (1 + MOMENTUM) / LR, 2),
           "init": {"top_eig": round(init_trace[-1], 3)}, "nets": {}}
    print(f"at init: top Hessian eigenvalue {init_trace[-1]:.2f}")

    for name, batch in BATCHES.items():
        model, rec = train_to_target(data, batch)
        guard.tick(f"train_{name}")
        theta = [p.detach().clone() for p in model.parameters()]
        d1 = filter_normalised_direction(model, seed=1)
        d2 = filter_normalised_direction(model, seed=2)
        rec["line_train"] = [round(loss_at(model, theta, [d1], [a], xs, ys), 4) for a in LINE]
        rec["line_test"] = [round(loss_at(model, theta, [d1], [a], xt, yt), 4) for a in LINE]
        rec["surface_train"] = [[round(loss_at(model, theta, [d1, d2], [a, b], xs, ys), 4) for a in GRID] for b in GRID]
        guard.tick(f"slices_{name}")
        trace = top_eigenvalue(model, xs, ys)
        rec["top_eig"] = round(trace[-1], 3)
        rec["top_eig_trace"] = [round(t, 3) for t in trace]
        rec["top_eig_rel_change_last10"] = round(abs(trace[-1] - trace[-11]) / abs(trace[-1]), 5)
        rec["weight_norm"] = round(torch.sqrt(sum((t * t).sum() for t in theta)).item(), 3)
        guard.tick(f"hessian_{name}")
        out["nets"][name] = rec
        print(f"{name:5s} batch {batch}: {rec['epochs']} epochs, {rec['steps']} steps, train loss {rec['train_loss']},"
              f" test acc {rec['test_acc']}, top eig {rec['top_eig']}, |theta| {rec['weight_norm']}")

    out["guard"] = guard.report()
    write_results("landscape", out)
    if args.figures:
        import figures
        figures.fig_landscape()


if __name__ == "__main__":
    main()
