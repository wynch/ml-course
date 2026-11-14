"""Section 5, second half: what a window saves, measured.

Two measurements, on untrained copies of module 04's GPT (weights change
neither memory nor time), with 4,096 learned positions so it can run that far:

1. **Generation.** Step through 4,096 tokens, batch 1, on the CPU. A full
   layer keeps a key and a value for every earlier token; a window layer
   keeps only the last w, in a rolling buffer (Mistral 7B's cache). We count
   the floats held after each token from the tensors themselves, and time
   each token.
2. **Training.** One forward and backward pass of one sequence of T tokens,
   T from 512 to 4,096, on the Apple GPU. Three ways to compute attention:
   the full causal mask; the window as a mask on the same T x T score matrix
   (what section 5's training does); and a banded form that computes only
   the scores inside the window, block by block. The banded form gives the
   same output as the masked one (the tests check it).

Run me:
    uv run python src/window_cost.py            # numbers, results/window_cost.json
    uv run python src/window_cost.py --figures  # also figures/window_cost.png
"""

from __future__ import annotations

import argparse
import math
import time
import types

from common import FIGURES, Guard, get_device, read_results, sync, write_results  # MPS guard first

import torch
import torch.nn.functional as F

from models import make_gpt
from window import set_windows

T_MAX = 4096
CHECK = (256, 1024, 4096)
TIME_WINDOW = 64            # tokens per timing bucket
W = 64                      # the window measured here
DECODE_ARMS = {"full": [None] * 3, f"w{W}": [W] * 3, f"mix{W}": [W, W, None]}
TRAIN_T = (512, 1024, 2048, 4096)
VOCAB = 65


class WindowCache:
    """One-token steps for module 04's GPT with a KV cache per layer.

    A layer with window w keeps at most its last w keys and values and drops
    the oldest as each new one comes in. A full layer keeps all of them.
    """

    def __init__(self, gpt, windows):
        self.gpt, self.windows = gpt, windows

    def init_state(self, batch: int, device):
        n = len(self.gpt.blocks)
        return {"pos": 0, "k": [None] * n, "v": [None] * n}

    @torch.no_grad()
    def step(self, idx_t, state):
        g, pos = self.gpt, state["pos"]
        x = g.wte(idx_t) + g.wpe.weight[pos]
        for i, (blk, w) in enumerate(zip(g.blocks, self.windows)):
            a = blk.attn
            B = x.shape[0]
            q, k, v = (t.view(B, a.n_head, 1, a.d_head) for t in a.qkv(blk.ln1(x)).split(a.d_model, dim=-1))
            K = k if state["k"][i] is None else torch.cat([state["k"][i], k], dim=2)
            V = v if state["v"][i] is None else torch.cat([state["v"][i], v], dim=2)
            if w is not None:
                K, V = K[:, :, -w:], V[:, :, -w:]
            state["k"][i], state["v"][i] = K, V
            att = torch.softmax((q @ K.transpose(-2, -1)) / math.sqrt(a.d_head), dim=-1)
            x = x + a.proj((att @ V).reshape(B, a.d_model))
            x = x + blk.mlp(blk.ln2(x))
        state["pos"] = pos + 1
        return g.head(g.ln_f(x)), state

    @staticmethod
    def state_floats(state) -> int:
        return sum(t.numel() for t in state["k"] + state["v"] if t is not None)


@torch.no_grad()
def walk(windows, device: str = "cpu") -> dict:
    torch.manual_seed(0)
    g = make_gpt(VOCAB, block_size=T_MAX).eval().to(device)
    s = WindowCache(g, windows)
    ids = torch.randint(0, VOCAB, (T_MAX,), generator=torch.Generator().manual_seed(0)).to(device)
    state = s.init_state(1, device)
    curve = {"pos": [], "ms_per_token": [], "state_floats": []}
    t0 = time.perf_counter()
    for t in range(T_MAX):
        _, state = s.step(ids[t:t + 1], state)
        if (t + 1) % TIME_WINDOW == 0:
            sync(device)
            now = time.perf_counter()
            curve["pos"].append(t + 1)
            curve["ms_per_token"].append(round(1000 * (now - t0) / TIME_WINDOW, 3))
            curve["state_floats"].append(s.state_floats(state))
            t0 = time.perf_counter()
    at = {str(c): {"state_floats": curve["state_floats"][curve["pos"].index(c)],
                   "tokens_per_s": round(1000 / curve["ms_per_token"][curve["pos"].index(c)], 1)} for c in CHECK}
    return {"curve": curve, "at": at}


# ------------------------------------------------------------------ banded attention for training
def banded_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Module 04's attention with a window of self.window, computing only the
    scores inside it. Cut the sequence into blocks of w; each block's queries
    read the keys of their own block and of the block before: 2w keys, not T.
    Cost O(T w) instead of O(T^2). Needs T to be a multiple of w."""
    B, T, C = x.shape
    w, H, dh = self.window, self.n_head, self.d_head
    assert T % w == 0, (T, w)
    nb = T // w
    q, k, v = (t.view(B, T, H, dh).transpose(1, 2) for t in self.qkv(x).split(self.d_model, dim=2))
    qb = q.reshape(B, H, nb, w, dh)
    kb, vb = k.reshape(B, H, nb, w, dh), v.reshape(B, H, nb, w, dh)
    pad = torch.zeros_like(kb[:, :, :1])
    k2 = torch.cat([torch.cat([pad, kb[:, :, :-1]], 2), kb], 3)              # (B, H, nb, 2w, dh)
    v2 = torch.cat([torch.cat([pad, vb[:, :, :-1]], 2), vb], 3)
    att = (qb @ k2.transpose(-2, -1)) / math.sqrt(dh)                         # (B, H, nb, w, 2w)
    i = torch.arange(w, device=x.device)[:, None]                            # query offset in its block
    j = torch.arange(2 * w, device=x.device)[None, :] - w                    # key offset from the block start
    ok = (j <= i) & (i - j < w)                                              # causal and inside the window
    first = ok & (j >= 0)                                                    # block 0 has no block before it
    mask = torch.stack([first] + [ok] * (nb - 1))                            # (nb, w, 2w)
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    y = (att @ v2).reshape(B, H, T, dh).transpose(1, 2).reshape(B, T, C)
    return self.proj(y)


def make_banded(gpt, w: int):
    for blk in gpt.blocks:
        blk.attn.window = w
        blk.attn.forward = types.MethodType(banded_forward, blk.attn)
    return gpt


def step_cost(kind: str, T: int, device: str, reps: int = 5) -> dict:
    """Median ms of one forward + backward on one sequence of T tokens, and the
    memory allocated between the start of the forward and its end (what the
    backward pass will need). The first pass warms up kernels and is not timed."""
    torch.manual_seed(0)
    g = make_gpt(VOCAB, block_size=T).to(device)
    if kind == "window_mask":
        set_windows(g, [W] * 3)
    elif kind == "banded":
        make_banded(g, W)
    x = torch.randint(0, VOCAB, (1, T), device=device)
    times, mem = [], 0
    loss = None
    for r in range(reps + 1):
        # Drop everything the last pass left behind, so the count below starts from the weights alone.
        g.zero_grad(set_to_none=True)
        for blk in g.blocks:
            blk.attn.last_attn = None
        loss = None
        sync(device)
        base = torch.mps.current_allocated_memory() if device == "mps" else 0
        t0 = time.perf_counter()
        _, loss = g(x, x)
        if device == "mps":
            sync(device)
            mem = torch.mps.current_allocated_memory() - base
        loss.backward()
        sync(device)
        if r:                                   # the first pass warms up kernels
            times.append(1000 * (time.perf_counter() - t0))
    del g, x, loss
    if device == "mps":
        torch.mps.empty_cache()
    times.sort()
    return {"ms": round(times[len(times) // 2], 2), "forward_alloc_bytes": int(mem)}


def run() -> dict:
    guard = Guard("window_cost")
    out = {"t_max": T_MAX, "window": W, "checkpoints": list(CHECK), "torch_threads": torch.get_num_threads(),
           "decode": {}, "train": {}}
    for name, windows in DECODE_ARMS.items():
        r = walk(windows, "cpu")
        out["decode"][name] = {"windows": windows} | r
        print("decode", name, {c: (v["state_floats"], v["tokens_per_s"]) for c, v in r["at"].items()})
        guard.tick(f"decode {name}")
    dev = get_device().type
    out["train_device"] = dev
    for kind in ("full", "window_mask", "banded"):
        out["train"][kind] = {}
        for T in TRAIN_T:
            out["train"][kind][str(T)] = step_cost(kind, T, dev)
            print("train", kind, T, out["train"][kind][str(T)])
            guard.tick()
        guard.tick(f"train {kind}")
    out["guard"] = guard.report()
    return out


def fig_cost(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    colors = {"full": "#453781", f"w{W}": "#22a884", f"mix{W}": "#d99a3e",
              "window_mask": "#22a884", "banded": "#b7410e"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    ax = axes[0]
    for n, v in r["decode"].items():
        ax.plot(v["curve"]["pos"], [4 * f / 1024 for f in v["curve"]["state_floats"]], color=colors[n], label=n)
    ax.set_xlabel("tokens so far")
    ax.set_ylabel("KB of keys and values held")
    ax.set_title("KV cache during generation", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    ax = axes[1]
    for n, v in r["decode"].items():
        ax.plot(v["curve"]["pos"], v["curve"]["ms_per_token"], color=colors[n], lw=1, label=n)
    ax.set_xlabel("tokens so far")
    ax.set_ylabel("ms per token (64-token mean)")
    ax.set_title("Time per generated token, CPU", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax = axes[2]
    labels = {"full": "full mask", "window_mask": f"window {W} as a mask", "banded": f"window {W}, banded"}
    for kind, v in r["train"].items():
        Ts = [int(t) for t in v]
        ax.plot(Ts, [v[str(t)]["ms"] for t in Ts], "o-", color=colors[kind], label=labels[kind])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("ms per forward + backward")
    ax.set_title(f"One training step, {r['train_device'].upper()}", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--redraw", action="store_true", help="only redraw the figure from results/window_cost.json")
    args = ap.parse_args()
    if args.redraw:
        fig_cost(read_results("window_cost"), FIGURES / "window_cost.png")
        return
    r = run()
    write_results("window_cost", r)
    if args.figures:
        fig_cost(r, FIGURES / "window_cost.png")


if __name__ == "__main__":
    main()
