"""Section 3, second half: what each model holds, and how fast it steps, as the context grows.

Generation is one token at a time. To produce token T+1, the GPT keeps a key
and a value for every earlier token in every layer (the KV cache); the GRU and
the SSM keep a fixed-size state. This script steps each model through 4,096
tokens, batch 1, and records at every position:

- the floats it holds between tokens, counted from the tensors themselves;
- on MPS, the change in allocated GPU memory since the start (a check on the count);
- the time per token, averaged over windows of 64 tokens.

The models are untrained, built at the trained models' sizes: weights change
neither the memory nor the time. The GPT gets 4,096 learned positions instead
of 128 (+507,904 parameters) so it can run that far. Each device runs the
three models one after the other while other programs share the Mac, so the
times carry noise; the memory counts do not.

Run me:
    uv run python src/decode.py            # numbers, results/decode.json
    uv run python src/decode.py --figures  # also figures/decode.png
"""

from __future__ import annotations

import argparse
import time

from common import FIGURES, Guard, sync, write_results  # MPS guard before torch

import torch

from models import GPTCache, build, make_gpt

T_MAX = 4096
CHECK = (256, 1024, 4096)
WINDOW = 64
KINDS = ("gpt", "gru", "selective")
VOCAB = 65


def stepper_for(kind: str):
    torch.manual_seed(0)
    if kind == "gpt":
        return GPTCache(make_gpt(VOCAB, block_size=T_MAX).eval())
    return build(kind, VOCAB).eval()


def module_of(s):
    return s.gpt if isinstance(s, GPTCache) else s


@torch.no_grad()
def walk(kind: str, device: str) -> dict:
    s = stepper_for(kind)
    module_of(s).to(device)
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (T_MAX,), generator=gen).to(device)
    sync(device)
    base = torch.mps.current_allocated_memory() if device == "mps" else 0
    state = s.init_state(1, device)
    curve = {"pos": [], "ms_per_token": [], "state_floats": [], "allocated_delta_bytes": []}
    t0 = time.perf_counter()
    for t in range(T_MAX):
        _, state = s.step(ids[t : t + 1], state)
        if (t + 1) % WINDOW == 0:
            sync(device)
            now = time.perf_counter()
            curve["pos"].append(t + 1)
            curve["ms_per_token"].append(round(1000 * (now - t0) / WINDOW, 3))
            curve["state_floats"].append(s.state_floats(state))
            if device == "mps":
                curve["allocated_delta_bytes"].append(torch.mps.current_allocated_memory() - base)
            t0 = time.perf_counter()
    at = {}
    for c in CHECK:
        i = curve["pos"].index(c)
        at[str(c)] = {
            "state_floats": curve["state_floats"][i],
            "state_bytes": 4 * curve["state_floats"][i],
            "tokens_per_s": round(1000 / curve["ms_per_token"][i], 1),
        }
    del s, state
    return {"curve": curve, "at": at}


def run() -> dict:
    guard = Guard("decode")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    out = {"t_max": T_MAX, "window": WINDOW, "checkpoints": list(CHECK), "torch_threads": torch.get_num_threads(),
           "devices": {}}
    for device in devices:
        out["devices"][device] = {}
        for kind in KINDS:
            r = walk(kind, device)
            out["devices"][device][kind] = r
            guard.tick(f"{device} {kind}")
            print(device, kind, {c: (v["state_bytes"], v["tokens_per_s"]) for c, v in r["at"].items()})
    out["guard"] = guard.report()
    return out


def fig_decode(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    colors = {"gpt": "#453781", "gru": "#d99a3e", "selective": "#b7410e"}
    labels = {"gpt": "GPT, KV cache", "gru": "GRU", "selective": "selective SSM"}
    devs = list(r["devices"])
    fig, axes = plt.subplots(1, 1 + len(devs), figsize=(4.2 * (1 + len(devs)), 3.8))
    ax = axes[0]
    for k in KINDS:
        c = r["devices"][devs[0]][k]["curve"]
        ax.plot(c["pos"], [4 * f / 1024 for f in c["state_floats"]], color=colors[k], label=labels[k])
    ax.set_yscale("log")
    ax.set_xlabel("tokens so far")
    ax.set_ylabel("KB held to make the next token")
    ax.set_title("Memory between tokens", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    for ax, d in zip(axes[1:], devs):
        for k in KINDS:
            c = r["devices"][d][k]["curve"]
            ax.plot(c["pos"], c["ms_per_token"], color=colors[k], lw=1, label=labels[k])
        ax.set_xlabel("tokens so far")
        ax.set_ylabel("ms per token (64-token mean)")
        ax.set_title(f"Time per token, {d.upper()}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    r = run()
    write_results("decode", r)
    if args.figures:
        fig_decode(r, FIGURES / "decode.png")


if __name__ == "__main__":
    main()
