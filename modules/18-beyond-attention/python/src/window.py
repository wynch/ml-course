"""Section 5, first half: sliding-window attention on module 04's GPT.

A window of w lets each token attend to itself and the w - 1 tokens before
it, nothing earlier. The parameters do not change: only the mask does. Six
arms, all module 04's GPT (3 layers, 4 heads, width 128, 619,776
parameters), trained with this module's recipe for 1,000 steps, two seeds:

    full    every layer sees all earlier tokens (module 04's mask)
    w64, w32, w16, w8    every layer sees the last w tokens
    mix8    two layers with w = 8, then one full layer: a local/global mix

Three measurements:

1. validation loss on tiny-shakespeare (the same 100 batches as sections 1-3);
2. the loss at each position of the 128-character window: a token at
   position t has t earlier characters to use, so the curve shows how much
   of that context each arm uses;
3. a copy task: 48 random letters, a bar, then the same 48 letters. Each
   letter of the second half is the one 48 positions back, so the model must
   reach exactly 48 tokens. One window reaches w - 1 tokens back; three
   stacked windows reach 3 (w - 1), but only by passing the letter on through
   tokens in between.

Run me:
    uv run python src/window.py            # numbers, results/window.json
    uv run python src/window.py --figures  # also figures/window.png
    uv run python src/window.py --long     # full against w8 for 4,000 steps, merged in
    uv run python src/window.py --checks   # review checks, results/window_checks.json, figures/window_long.png
"""

from __future__ import annotations

import argparse
import math

from common import BLOCK, FIGURES, CharData, Guard, get_device, read_results, write_results  # MPS guard first

import torch
import torch.nn.functional as F

from models import make_gpt, n_params
from train import train

STEPS = 1000
SEEDS = (0, 1)
ARMS = {
    "full": [None, None, None],
    "w64": [64, 64, 64],
    "w32": [32, 32, 32],
    "w16": [16, 16, 16],
    "w8": [8, 8, 8],
    "mix8": [8, 8, None],
}
COPY_LEN = 48                      # letters per half; every target sits COPY_LEN tokens back
COPY_VOCAB = 27                    # a-z and the bar


def window_mask(T: int, w: int | None) -> torch.Tensor:
    """(T, T) bool: query t may read key s when s <= t and t - s < w. None: plain causal."""
    t = torch.arange(T)[:, None]
    s = torch.arange(T)[None, :]
    ok = s <= t
    return ok if w is None else ok & (t - s < w)


def set_windows(gpt, windows) -> None:
    """Overwrite each block's causal mask in place with its window. Module 04's
    attention fills every 0 in its ``mask`` buffer with -inf before the softmax."""
    T = gpt.cfg.block_size
    for blk, w in zip(gpt.blocks, windows):
        blk.attn.mask.copy_(window_mask(T, w).float().view(1, 1, T, T))


def reach(windows) -> float:
    """How far back the last layer can see, through the stack: sum of (w - 1). inf if any layer is full."""
    return math.inf if any(w is None for w in windows) else sum(w - 1 for w in windows)


def make(windows, vocab: int, seed: int):
    torch.manual_seed(seed)
    g = make_gpt(vocab)
    set_windows(g, windows)
    return g


@torch.no_grad()
def loss_by_position(model, windows, device) -> list[float]:
    """Mean loss at each of the 128 positions, over the 100 fixed validation batches."""
    model.eval()
    tot = torch.zeros(BLOCK)
    for x, y in windows:
        logits, _ = model(x.to(device))
        ce = F.cross_entropy(logits.transpose(1, 2), y.to(device), reduction="none")   # (B, T)
        tot += ce.mean(0).cpu()
    return [round(float(v), 4) for v in tot / len(windows)]


# ------------------------------------------------------------------ the copy task
class CopyData:
    """48 random letters, '|', the same 48 letters. Loss counts only the second half."""

    def batch(self, split: str, gen: torch.Generator, device, batch: int = 64, block: int = 0):
        s = torch.randint(0, 26, (batch, COPY_LEN), generator=gen)
        bar = torch.full((batch, 1), 26)
        seq = torch.cat([s, bar, s], 1)                     # (B, 97)
        x, y = seq[:, :-1].clone(), seq[:, 1:].clone()      # 96 inputs
        y[:, :COPY_LEN] = -100                              # first half and the bar: not scored
        return x.to(device), y.to(device)

    def val_windows(self, n: int = 20, batch: int = 64, block: int = 0):
        gen = torch.Generator().manual_seed(12345)
        return [self.batch("val", gen, "cpu", batch) for _ in range(n)]


def copy_distance() -> int:
    """How far back each scored target sits. At input position p (48..95) the
    target is seq[p + 1] = s[p - 48], which is the input at position p - 48."""
    x, y = CopyData().batch("val", torch.Generator().manual_seed(0), "cpu", 4)
    p = torch.arange(COPY_LEN, 2 * COPY_LEN)
    assert torch.equal(y[:, p], x[:, p - COPY_LEN])
    return COPY_LEN


@torch.no_grad()
def copy_score(model, data: CopyData, device) -> dict:
    model.eval()
    loss, hit, n = 0.0, 0, 0
    for x, y in data.val_windows():
        logits, l = model(x.to(device), y.to(device))
        m = y != -100
        loss += l.item()
        hit += int((logits.argmax(-1).cpu()[m] == y[m]).sum())
        n += int(m.sum())
    return {"loss": round(loss / 20, 4), "accuracy": round(hit / n, 4)}


def run() -> dict:
    guard = Guard("window")
    device = get_device()
    data = CharData()
    vq, vf = data.val_windows(10), data.val_windows(100)
    out = {"steps": STEPS, "seeds": list(SEEDS), "arms": {}, "copy": {"distance": copy_distance(), "arms": {}}}
    for name, windows in ARMS.items():
        runs = []
        for seed in SEEDS:
            g = make(windows, data.vocab_size, seed)
            r = train(g, data, device, steps=STEPS, seed=seed, val_quick=vq, val_full=vf)
            r["by_position"] = loss_by_position(g, vf, device)
            runs.append({k: r[k] for k in ("steps", "ms_per_step", "val_loss", "by_position", "log")} | {"seed": seed})
            print(f"  {name:5s} seed {seed}: val {r['val_loss']:.4f}  {r['ms_per_step']} ms/step")
            del g
            guard.tick(f"{name} seed {seed}")
        out["arms"][name] = {"windows": windows, "reach": reach(windows) if not math.isinf(reach(windows)) else None,
                             "params": n_params(make(windows, data.vocab_size, 0)), "runs": runs}
    cd = CopyData()
    for name, windows in ARMS.items():
        g = make(windows, COPY_VOCAB, 0)
        r = train(g, cd, device, steps=STEPS, seed=0, val_quick=cd.val_windows(4), val_full=cd.val_windows())
        s = copy_score(g, cd, device)
        out["copy"]["arms"][name] = s | {"curve": r["log"]["val"], "curve_step": r["log"]["val_step"]}
        print(f"  copy {name:5s}: loss {s['loss']:.3f}  accuracy {s['accuracy']:.3f}")
        del g
        guard.tick(f"copy {name}")
    out["copy"]["chance_loss"] = round(math.log(26), 4)
    out["guard"] = guard.report()
    return out


LONG_STEPS = 4000
LONG_ARMS = ("full", "w8")


def run_long() -> dict:
    """A check added after the main run: full against w8 for 4,000 steps, seed 0.
    Does the window's lead hold, or is it a head start?"""
    guard = Guard("window --long")
    device = get_device()
    data = CharData()
    vq, vf = data.val_windows(10), data.val_windows(100)
    out = {"steps": LONG_STEPS, "seed": 0, "arms": {}}
    for name in LONG_ARMS:
        g = make(ARMS[name], data.vocab_size, 0)
        r = train(g, data, device, steps=LONG_STEPS, seed=0, val_quick=vq, val_full=vf, eval_every=0.05)
        out["arms"][name] = {"val_loss": r["val_loss"], "ms_per_step": r["ms_per_step"],
                             "by_position": loss_by_position(g, vf, device),
                             "val_step": r["log"]["val_step"], "val": r["log"]["val"]}
        print(f"  long {name:5s}: val {r['val_loss']:.4f}")
        del g
        guard.tick(f"long {name}")
    out["guard"] = guard.report()
    return out


CHECK_SEEDS = (2, 3)                 # two more seeds at 1,000 steps
CHECK_ARMS = ("full", "w8", "mix8")
LONG_SEEDS = (0, 1, 2)


def run_checks() -> dict:
    """Checks added in review, written to results/window_checks.json:
    1. full, w8 and mix8 at seeds 2 and 3, 1,000 steps;
    2. full and w8 for 4,000 steps at seeds 0, 1 and 2, with training and validation curves;
    3. the copy task for w32 and full, then each layer's attention switched off in turn."""
    guard = Guard("window --checks")
    device = get_device()
    data = CharData()
    vq, vf = data.val_windows(10), data.val_windows(100)
    out = {"short": {}, "long": {"steps": LONG_STEPS, "arms": {}}, "copy_ablation": {}}
    for name in CHECK_ARMS:
        out["short"][name] = []
        for seed in CHECK_SEEDS:
            g = make(ARMS[name], data.vocab_size, seed)
            r = train(g, data, device, steps=STEPS, seed=seed, val_quick=vq, val_full=vf)
            out["short"][name].append({"seed": seed, "val_loss": r["val_loss"]})
            print(f"  {name:5s} seed {seed}: val {r['val_loss']:.4f}")
            guard.tick(f"{name} seed {seed}")
    for name in LONG_ARMS:
        out["long"]["arms"][name] = []
        for seed in LONG_SEEDS:
            g = make(ARMS[name], data.vocab_size, seed)
            r = train(g, data, device, steps=LONG_STEPS, seed=seed, val_quick=vq, val_full=vf, eval_every=0.05)
            out["long"]["arms"][name].append({"seed": seed, "val_loss": r["val_loss"],
                                              "by_position": loss_by_position(g, vf, device)} | r["log"])
            print(f"  long {name:5s} seed {seed}: val {r['val_loss']:.4f}")
            guard.tick(f"long {name} seed {seed}")
    cd = CopyData()
    for name in ("w32", "full"):
        g = make(ARMS[name], COPY_VOCAB, 0)
        train(g, cd, device, steps=STEPS, seed=0, val_quick=cd.val_windows(4), val_full=cd.val_windows())
        row = {"intact": copy_score(g, cd, device)}
        for i, blk in enumerate(g.blocks):
            keep = blk.attn.forward
            blk.attn.forward = torch.zeros_like             # this layer's attention adds nothing
            row[f"layer {i + 1} attention off"] = copy_score(g, cd, device)
            blk.attn.forward = keep
        out["copy_ablation"][name] = row
        print(f"  copy {name}: {row}")
        guard.tick(f"copy {name}")
    out["guard"] = guard.report()
    return out


def fig_long(c: dict, path) -> None:
    """Full against w8 over 4,000 steps, three seeds: validation (solid) and training loss (dotted)."""
    import matplotlib.pyplot as plt

    colors = {"full": "#453781", "w8": "#d99a3e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for name, runs in c["long"]["arms"].items():
        for i, run in enumerate(runs):
            lab = name if i == 0 else None
            axes[0].plot(run["val_step"], run["val"], color=colors[name], lw=1.2, label=lab)
            axes[0].plot(run["step"], run["train"], color=colors[name], lw=0.8, ls=":")
    axes[0].set_ylim(1.0, 2.4)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (nats); solid validation, dotted training")
    axes[0].set_title("4,000 steps, three seeds", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(alpha=0.25)
    ax = axes[1]
    f, w = c["long"]["arms"]["full"], c["long"]["arms"]["w8"]
    for a, b, shade in zip(f, w, ("0.15", "0.45", "0.7")):
        ax.plot(a["val_step"], [x - y for x, y in zip(a["val"], b["val"])], color=shade, lw=1.2,
                label=f"seed {a['seed']}")
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss, full minus w8")
    ax.set_title("Above 0: the window is ahead", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_window(r: dict, path) -> None:
    import matplotlib.pyplot as plt

    names = list(r["arms"])
    colors = dict(zip(names, ["#453781", "#2a788e", "#22a884", "#7ad151", "#d99a3e", "#b7410e"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    for i, n in enumerate(names):
        vals = [run["val_loss"] for run in r["arms"][n]["runs"]]
        ax.plot([i] * len(vals), vals, "o", color=colors[n])
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("validation loss (nats), two seeds")
    ax.set_title("Tiny-shakespeare, 1,000 steps", fontsize=10)
    ax.grid(alpha=0.25)
    ax = axes[1]
    for n in names:
        bp = [sum(v) / len(v) for v in zip(*(run["by_position"] for run in r["arms"][n]["runs"]))]
        ax.plot(range(len(bp)), bp, color=colors[n], lw=1.2, label=n)
    ax.set_xlabel("position in the 128-character window")
    ax.set_ylabel("validation loss at that position")
    ax.set_ylim(1.35, 2.2)
    ax.set_title("Loss by position (mean of two seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[2]
    for n in names:
        c = r["copy"]["arms"][n]
        ax.plot(c["curve_step"], c["curve"], color=colors[n], label=f"{n}: {c['loss']:.2f}")
    ax.axhline(r["copy"]["chance_loss"], color="grey", ls=":", lw=1)
    ax.text(10, r["copy"]["chance_loss"] + 0.05, "chance, ln 26", fontsize=8, color="grey")
    ax.set_xlabel("step")
    ax.set_ylabel("copy loss, second half (nats)")
    ax.set_title(f"Copy from {r['copy']['distance']} tokens back", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--redraw", action="store_true", help="only redraw the figure from results/window.json")
    ap.add_argument("--long", action="store_true", help="full against w8 for 4,000 steps, merged into results/window.json")
    ap.add_argument("--checks", action="store_true",
                    help="review checks: more seeds, three 4,000-step pairs, copy ablation; results/window_checks.json")
    args = ap.parse_args()
    if args.redraw:
        fig_window(read_results("window"), FIGURES / "window.png")
        fig_long(read_results("window_checks"), FIGURES / "window_long.png")
        return
    if args.checks:
        c = run_checks()
        write_results("window_checks", c)
        fig_long(c, FIGURES / "window_long.png")
        return
    if args.long:
        r = read_results("window")
        r["long"] = run_long()
        write_results("window", r)
        if args.figures:
            fig_window(r, FIGURES / "window.png")
        return
    r = run()
    write_results("window", r)
    if args.figures:
        fig_window(r, FIGURES / "window.png")


if __name__ == "__main__":
    main()
