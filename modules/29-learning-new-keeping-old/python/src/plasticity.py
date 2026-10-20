"""Section 4: losing the ability to learn. One small network learns a long
row of tasks, each FashionMNIST with its pixels shuffled a new way. Does each
new task go as well as the first? Plain backprop against continual backprop
(Dohare et al., 2024), which keeps resetting a few of its least useful units.

    uv run python src/plasticity.py          # repeat until it prints "remaining: 0"

Each call stops after about four minutes; a run saves its state between calls.
CPU, 4 threads.
"""

from __future__ import annotations

import argparse
import math
import time

import common  # noqa: F401  (laptop guard before torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FASHION, MODULE, RESULTS, Guard, parquet, write_results

CACHE = MODULE / "python" / "outputs" / "fashion_mnist.npz"
RUNS = RESULTS / "plasticity"
CHUNK_SECONDS = 240

HIDDEN = (256, 256)
TASKS = 800
PER_TASK = 6000            # training images per task, one pass
BATCH = 32
TEST = 2000                # test images scored after each task
SEEDS = (0, 1, 2)
ARMS = {
    "backprop": dict(opt="adam", lr=1e-3, cbp=False),
    "cbp": dict(opt="adam", lr=1e-3, cbp=True),
    "sgd": dict(opt="sgd", lr=0.1, cbp=False),
    "fresh": dict(opt="adam", lr=1e-3, cbp=False, fresh=True, tasks=40),
}
# Continual backprop, Dohare et al. (2024): decay of the running utility,
# share of units replaced per step, steps a unit is protected after a reset.
CBP_DECAY = 0.99
CBP_RATE = 1e-4
CBP_MATURITY = 100


def fashion() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not CACHE.exists():
        import io

        from PIL import Image

        def arrays(split):
            rows = parquet(FASHION, f"fashion_mnist/{split}-00000-of-00001.parquet")
            x = np.stack([np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])), dtype=np.uint8) for r in rows])
            return x.reshape(len(rows), -1), np.array([r["label"] for r in rows], dtype=np.int64)

        CACHE.parent.mkdir(parents=True, exist_ok=True)
        xtr, ytr = arrays("train")
        xte, yte = arrays("test")
        np.savez_compressed(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    z = np.load(CACHE)
    return z["xtr"], z["ytr"], z["xte"], z["yte"]


class Net(nn.Module):
    def __init__(self, sizes=(784, *HIDDEN, 10)):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        for lin in self.layers:
            init_unit_in(lin, torch.arange(lin.out_features))
            nn.init.zeros_(lin.bias)

    def forward(self, x, keep: bool = False):
        hs = []
        for lin in self.layers[:-1]:
            x = F.relu(lin(x))
            hs.append(x)
        out = self.layers[-1](x)
        return (out, hs) if keep else out


def make_opt(net: nn.Module, spec: dict) -> torch.optim.Optimizer:
    if spec["opt"] == "adam":
        return torch.optim.Adam(net.parameters(), lr=spec["lr"])
    return torch.optim.SGD(net.parameters(), lr=spec["lr"])


def init_unit_in(lin: nn.Linear, units: torch.Tensor) -> None:
    """Kaiming-uniform input weights for the chosen units, as at the start."""
    bound = math.sqrt(6.0 / lin.in_features)
    with torch.no_grad():
        lin.weight[units] = torch.empty(len(units), lin.in_features).uniform_(-bound, bound)
        lin.bias[units] = 0.0


class CBP:
    """Continual backprop's generate-and-test: track each hidden unit's
    contribution utility, |activation| x sum of |outgoing weights|, as a running
    average; every step, replace a few of the lowest-utility mature units with
    fresh ones (new input weights, zero outgoing weights)."""

    def __init__(self, net: Net, opt=None, rate=CBP_RATE, decay=CBP_DECAY, maturity=CBP_MATURITY):
        self.net, self.opt, self.rate, self.decay, self.maturity = net, opt, rate, decay, maturity
        self.util = [torch.zeros(n) for n in HIDDEN]
        self.age = [torch.zeros(n, dtype=torch.long) for n in HIDDEN]
        self.owed = [0.0 for _ in HIDDEN]
        self.replaced = 0

    @torch.no_grad()
    def step(self, hs: list[torch.Tensor]) -> None:
        for i, h in enumerate(hs):
            out_w = self.net.layers[i + 1].weight.abs().sum(0)
            self.util[i].mul_(self.decay).add_((1 - self.decay) * h.abs().mean(0) * out_w)
            self.age[i] += 1
            self.owed[i] += self.rate * len(self.util[i])
            mature = (self.age[i] > self.maturity).nonzero().squeeze(1)
            n = min(int(self.owed[i]), len(mature))
            if n == 0:
                continue
            self.owed[i] -= n
            worst = mature[self.util[i][mature].argsort()[:n]]
            init_unit_in(self.net.layers[i], worst)
            self.net.layers[i + 1].weight[:, worst] = 0.0
            for p, sl in ((self.net.layers[i].weight, worst), (self.net.layers[i].bias, worst),
                          (self.net.layers[i + 1].weight, (slice(None), worst))):
                for v in (self.opt.state.get(p, {}) if self.opt else {}).values():
                    if torch.is_tensor(v) and v.shape == p.shape:
                        v[sl] = 0.0          # a fresh unit starts with fresh Adam moments
            self.util[i][worst] = 0.0
            self.age[i][worst] = 0
            self.replaced += n


@torch.no_grad()
def health(net: Net, x: torch.Tensor) -> dict:
    """Three signs of lost plasticity on 1,000 inputs: dead units, weight size, effective rank."""
    _, hs = net(x, keep=True)
    dead = [float((h.max(0).values <= 0).float().mean()) for h in hs]
    s = torch.linalg.svdvals(hs[-1].double())
    p = s / s.sum().clamp_min(1e-12)       # all-zero features: rank 0, not 0/0
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())) if s.sum() > 0 else 0.0
    wmag = float(torch.cat([lin.weight.abs().flatten() for lin in net.layers]).mean())
    return {"dead": round(float(np.mean(dead)), 4), "erank": round(erank, 2), "weight": round(wmag, 5)}


def run(arm: str, seed: int, budget: float) -> bool:
    """Advance one run by up to `budget` seconds; True when it has finished."""
    spec = ARMS[arm]
    tasks = spec.get("tasks", TASKS)
    path = RUNS / f"{arm}_s{seed}.json"
    state_path = MODULE / "python" / "outputs" / f"plasticity_{arm}_s{seed}.pt"
    xtr, ytr, xte, yte = fashion()
    xtr = torch.from_numpy(xtr).float() / 255.0
    xte = torch.from_numpy(xte).float() / 255.0
    ytr, yte = torch.from_numpy(ytr), torch.from_numpy(yte)
    if state_path.exists():
        st = torch.load(state_path, weights_only=False)
    else:
        torch.manual_seed(seed)
        net = Net()
        st = {"net": net, "opt": make_opt(net, spec), "cbp": None, "task": 0, "log": [], "seconds": 0.0,
              "rng": np.random.default_rng(seed), "torch_rng": torch.get_rng_state()}
        st["cbp"] = CBP(net, st["opt"]) if spec["cbp"] else None
    torch.set_rng_state(st["torch_rng"])
    net, opt, cbp, rng = st["net"], st["opt"], st["cbp"], st["rng"]
    t0 = time.perf_counter()
    while st["task"] < tasks and time.perf_counter() - t0 < budget:
        if spec.get("fresh"):
            net = Net()
            opt = make_opt(net, spec)
        perm = torch.from_numpy(rng.permutation(784))
        idx = torch.from_numpy(rng.choice(len(xtr), PER_TASK, replace=False))
        hits, seen = 0, 0
        for b in range(0, PER_TASK, BATCH):
            j = idx[b:b + BATCH]
            out, hs = net(xtr[j][:, perm], keep=True)
            loss = F.cross_entropy(out, ytr[j])
            hits += int((out.argmax(1) == ytr[j]).sum()); seen += len(j)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if cbp is not None:
                cbp.step([h.detach() for h in hs])
        te = torch.from_numpy(rng.choice(len(xte), TEST, replace=False))
        with torch.no_grad():
            acc = float((net(xte[te][:, perm]).argmax(1) == yte[te]).float().mean())
        row = {"task": st["task"], "test_acc": round(acc, 4), "online_acc": round(hits / seen, 4)}
        if st["task"] % 10 == 0 or st["task"] == tasks - 1:
            row |= health(net, xte[te[:1000]][:, perm])
        st["log"].append(row)
        st["task"] += 1
    st["net"], st["opt"] = net, opt
    st["seconds"] += time.perf_counter() - t0
    st["torch_rng"] = torch.get_rng_state()
    if st["task"] < tasks:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st, state_path)
        print(f"  {arm} s{seed}: task {st['task']} of {tasks}")
        return False
    write_results(path.stem, {
        "arm": arm, "seed": seed, "spec": spec | {"tasks": tasks}, "hidden": list(HIDDEN),
        "per_task": PER_TASK, "batch": BATCH, "test_per_task": TEST,
        "cbp": {"decay": CBP_DECAY, "rate": CBP_RATE, "maturity": CBP_MATURITY,
                "replaced": cbp.replaced} if cbp else None,
        "log": st["log"], "seconds": round(st["seconds"], 1),
        "peak_rss_gb": round(Guard.rss_peak_bytes() / 2**30, 2),
    }, RUNS)
    state_path.unlink(missing_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [(a, s) for s in SEEDS for a in ARMS if not (RUNS / f"{a}_s{s}.json").exists()]
    if args.arm:
        todo = [(a, s) for a, s in todo if a == args.arm and (args.seed is None or s == args.seed)]
    t0 = time.perf_counter()
    while todo:
        left = CHUNK_SECONDS - (time.perf_counter() - t0)
        if left < 10:
            break
        arm, seed = todo[0]
        if run(arm, seed, left):
            todo.pop(0)
    print(f"remaining: {len(todo)}")


if __name__ == "__main__":
    main()
