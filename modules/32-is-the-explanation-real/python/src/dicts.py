"""Fit SAEs and transcoders to block 1's MLP, on the trained model and on its random copy.

Both dictionaries have the true MLP's own shape, 2,048 latents with TopK k = 8:

    SAE:         z = TopK(ReLU(W_enc (y - b_pre) + b_enc)),  y' = W_dec z + b_dec      (rebuilds the MLP output y)
    transcoder:  z = TopK(ReLU(W_enc (x - b_pre) + b_enc)),  y' = W_dec z + b_dec      (maps the MLP input x to y)

Decoder columns are held at unit norm, as in module 12. Inputs and targets
are scaled to a mean squared norm of 128 while training (the random model's
MLP output is about a hundred times smaller than the trained model's), and
the scales are folded back into the weights afterwards. Loss: squared error
divided by the variance of y. Data: block 1's MLP input (after LayerNorm) and
output on the first 2^20 tokens of the training stories, recorded fresh each
run (about 10 s). Seeds 0, 1 and 2 for each kind and each model: twelve
dictionaries, saved to ``artifacts/runs/dict_<model>_<kind>_s<seed>.pt``.

Each call trains the dictionaries still missing until about 4.5 minutes have
passed, then exits; run it again until it prints "done".

    uv run python src/dicts.py
    uv run python src/dicts.py --long      # twice: seed 0 of each kind at 10,000 steps (added after the first results)
"""

from __future__ import annotations

import time

from common import ARTIFACTS, CHUNK_SECONDS, RUNS, Guard, append_log, free, get_device

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import Config, GPT, load, random_copy

LAYER = 1
N_LAT, K = 2048, 8
STEPS, BATCH, LR = 3000, 4096, 1e-3
N_TOKENS = 2**20
SEEDS = (0, 1, 2)
MODELS = ("trained", "random")
KINDS = ("sae", "transcoder")


class TopKDict(torch.nn.Module):
    def __init__(self, d: int, n: int, k: int, b_pre: torch.Tensor, b_dec: torch.Tensor):
        super().__init__()
        self.k = k
        w = torch.randn(d, n)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(w)                 # (d, n), unit columns
        self.W_enc = torch.nn.Parameter(w.T.clone())       # (n, d), tied at init
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_pre = torch.nn.Parameter(b_pre.clone())
        self.b_dec = torch.nn.Parameter(b_dec.clone())

    def encode(self, x):
        pre = F.relu((x - self.b_pre) @ self.W_enc.T + self.b_enc)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def decode(self, z):
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm(self):
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)


def get_model(which: str, device) -> GPT:
    if which == "trained":
        return load(ARTIFACTS / "sparse.pt", device)
    return random_copy(Config()).to(device)


@torch.no_grad()
def record(model: GPT, ids: np.ndarray, device, n_tokens: int = N_TOKENS, layer: int = LAYER):
    """Block ``layer``'s MLP input (after LayerNorm) and output, (n_tokens, 128) each.

    Written into two preallocated tensors, 64 windows at a time, so the peak stays near their size (1 GB)."""
    ctx = model.c.ctx
    n = n_tokens // ctx
    x = torch.from_numpy(ids[: n * ctx].reshape(n, ctx))
    ins = torch.empty(n * ctx, model.c.d, device=device)
    outs = torch.empty(n * ctx, model.c.d, device=device)
    at = [0]
    blk = model.blocks[layer]

    def grab_in(m, i, o):
        ins[at[0] : at[0] + o.shape[0] * o.shape[1]] = o.flatten(0, 1)

    def grab_out(m, i, o):
        outs[at[0] : at[0] + o.shape[0] * o.shape[1]] = o.flatten(0, 1)
        at[0] += o.shape[0] * o.shape[1]

    h1 = blk.ln2.register_forward_hook(grab_in)
    h2 = blk.mlp.register_forward_hook(grab_out)
    try:
        for s in range(0, n, 64):
            model(x[s : s + 64].to(device))
    finally:
        h1.remove(); h2.remove()
    return ins, outs


def train_dict(x: torch.Tensor, y: torch.Tensor, seed: int, steps: int = STEPS) -> tuple[TopKDict, list]:
    """Train on x and y each scaled to mean squared norm d, then fold the two scales into the weights,
    so the returned dictionary reads and writes raw activations (TopK and ReLU ignore a positive scale)."""
    sx = float(x[:65536].pow(2).sum(-1).mean().div(x.shape[1]).sqrt())
    sy = float(y[:65536].pow(2).sum(-1).mean().div(y.shape[1]).sqrt())
    d, curve = _train(x / sx, y / sy, seed, steps)
    with torch.no_grad():
        d.W_enc.mul_(sy / sx)
        d.b_enc.mul_(sy)
        d.b_pre.mul_(sx)
        d.b_dec.mul_(sy)
    return d, curve


def _train(x: torch.Tensor, y: torch.Tensor, seed: int, steps: int) -> tuple[TopKDict, list]:
    torch.manual_seed(seed)
    dev = x.device
    d = TopKDict(x.shape[1], N_LAT, K, x[:65536].mean(0), y[:65536].mean(0)).to(dev)
    opt = torch.optim.Adam(d.parameters(), lr=LR)
    var = ((y[:65536] - y[:65536].mean(0)) ** 2).sum(-1).mean()
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for step in range(steps):
        idx = torch.randint(0, len(x), (BATCH,), generator=gen).to(dev)
        yh, _ = d(x[idx])
        loss = ((y[idx] - yh) ** 2).sum(-1).mean() / var
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():                    # drop the radial part; renorm() holds the norm at 1
            g = d.W_dec.grad
            g -= d.W_dec * (g * d.W_dec).sum(0, keepdim=True)
        opt.step()
        d.renorm()
        if step % 100 == 0 or step == steps - 1:
            curve.append([step, round(float(loss.detach()), 5)])
    return d.eval(), curve


def path_for(model: str, kind: str, seed: int | str):
    return RUNS / f"dict_{model}_{kind}_s{seed}.pt"


LONG_STEPS = 10000          # added after the first results: is the transcoder's gap a training budget?


def run_long() -> None:
    """Seed 0 of each kind on the trained model, 10,000 steps instead of 3,000; one dictionary per call."""
    guard = Guard("dicts.py --long")
    dev = get_device()
    tr, _ = data.tokens()
    todo = [k for k in KINDS if not path_for("trained", k, "long").exists()]
    if not todo:
        print("done")
        return
    kind = todo[0]
    model = get_model("trained", dev)
    x, y = record(model, tr, dev)
    del model
    ts = time.perf_counter()
    d, curve = train_dict(y if kind == "sae" else x, y, 0, steps=LONG_STEPS)
    torch.save({"state": {k: v.cpu() for k, v in d.state_dict().items()}, "curve": curve,
                "seconds": round(time.perf_counter() - ts, 1), "steps": LONG_STEPS, "batch": BATCH, "lr": LR,
                "n_latents": N_LAT, "k": K, "tokens": N_TOKENS}, path_for("trained", kind, "long"))
    print(f"trained {kind} long: final nmse {curve[-1][1]} ({time.perf_counter() - ts:.0f} s)")
    g = guard.report()
    append_log("dicts", {"trained": [f"trained/{kind}/long"], "seconds": g["seconds"], "peak_rss_gb": g["peak_rss_gb"],
                         "peak_mps_driver_gb": g["peak_mps_driver_gb"]})
    print("done" if len(todo) == 1 else f"{len(todo) - 1} left")


def run() -> None:
    guard = Guard("dicts.py")
    dev = get_device()
    tr, _ = data.tokens()
    t0 = time.perf_counter()
    done_now = []
    for which in MODELS:
        todo = [(k, s) for k in KINDS for s in SEEDS if not path_for(which, k, s).exists()]
        if not todo:
            continue
        model = get_model(which, dev)
        x, y = record(model, tr, dev)
        del model
        guard.tick(f"record {which}")
        for kind, seed in todo:
            if time.perf_counter() - t0 > CHUNK_SECONDS - 40:
                break
            ts = time.perf_counter()
            d, curve = train_dict(y if kind == "sae" else x, y, seed)
            torch.save({"state": {k: v.cpu() for k, v in d.state_dict().items()}, "curve": curve,
                        "seconds": round(time.perf_counter() - ts, 1), "steps": STEPS, "batch": BATCH, "lr": LR,
                        "n_latents": N_LAT, "k": K, "tokens": N_TOKENS}, path_for(which, kind, seed))
            done_now.append(f"{which}/{kind}/s{seed}")
            print(f"{which} {kind} seed {seed}: final nmse {curve[-1][1]} ({time.perf_counter() - ts:.0f} s)")
            guard.tick(f"{which} {kind} s{seed}")
        del x, y
        free()
    g = guard.report()
    append_log("dicts", {"trained": done_now, "seconds": g["seconds"], "peak_rss_gb": g["peak_rss_gb"],
                         "peak_mps_driver_gb": g["peak_mps_driver_gb"]})
    left = [1 for m in MODELS for k in KINDS for s in SEEDS if not path_for(m, k, s).exists()]
    print("done" if not left else f"{len(left)} left")


if __name__ == "__main__":
    import sys

    run_long() if "--long" in sys.argv else run()
