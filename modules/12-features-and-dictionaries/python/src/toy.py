"""Section 1: the toy model of superposition, from scratch, on the CPU.

The model from Elhage et al., "Toy Models of Superposition" (2022):

    h  = W x                  x has n features, h has m < n dimensions
    x' = ReLU(W^T h + b)      read the n features back out
    loss = sum_i I_i (x_i - x'_i)^2

Each feature is zero with probability S (the sparsity) and uniform on [0, 1]
otherwise. I_i is the feature's importance. Column W_i is the direction the
model gives feature i in the m-dimensional space.

Two setups:
  A. n = 20 features, m = 5 dimensions, importance 0.7^i (the paper's main one).
  B. n = 5 features, m = 2 dimensions, importance 0.9^i, so every direction
     can be drawn on the page.

For each sparsity S we train three seeds and keep the one with the lowest
loss. A feature counts as represented when its column has norm above 0.5.

This is a toy. It shows that a ReLU readout CAN pack more sparse features than
it has dimensions. It says nothing about whether real networks do.

Run me:
    uv run python src/toy.py            # numbers, results/toy.json
    uv run python src/toy.py --figures  # also figures/toy_superposition.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, write_results  # sets the MPS guard before torch

import numpy as np
import torch

SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
SEEDS = (0, 1, 2)
STEPS = 10000
BATCH = 1024
LR = 1e-3
REPRESENTED = 0.5          # a feature is represented when ||W_i|| > 0.5
SETUPS = {
    "A": {"n": 20, "m": 5, "importance_decay": 0.7},
    "B": {"n": 5, "m": 2, "importance_decay": 0.9},
}


def sample(batch: int, n: int, sparsity: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """(instances, batch, n): each feature is 0 with prob S, else uniform on [0, 1]."""
    k = sparsity.shape[0]
    x = torch.rand(k, batch, n, generator=gen)
    on = torch.rand(k, batch, n, generator=gen) >= sparsity[:, None, None]
    return x * on


def forward(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x' = ReLU(W^T W x + b) for k models at once: W (k, m, n), b (k, n), x (k, batch, n)."""
    h = torch.einsum("kmn,kbn->kbm", W, x)                    # squeeze into m dimensions
    return torch.relu(torch.einsum("kmn,kbm->kbn", W, h) + b[:, None, :])


def loss_fn(W, b, x, importance) -> torch.Tensor:
    """Importance-weighted squared error, one value per model: (k,)."""
    return (importance * (x - forward(W, b, x)) ** 2).sum(-1).mean(-1)


def train(n: int, m: int, importance: torch.Tensor, sparsities: list[float], seed: int,
          steps: int = STEPS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train one toy model per sparsity in parallel. Returns W (k, m, n), b (k, n), final loss (k,)."""
    gen = torch.Generator().manual_seed(seed)
    k = len(sparsities)
    W = torch.nn.Parameter(torch.empty(k, m, n))
    torch.nn.init.xavier_normal_(W, generator=gen)
    b = torch.nn.Parameter(torch.zeros(k, n))
    S = torch.tensor(sparsities)
    opt = torch.optim.Adam([W, b], lr=LR)
    for step in range(steps):
        x = sample(BATCH, n, S, gen)
        loss = loss_fn(W, b, x, importance)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    # A fresh, larger batch for the reported loss.
    with torch.no_grad():
        final = loss_fn(W, b, sample(8 * BATCH, n, S, gen), importance)
    return W.detach(), b.detach(), final


def describe(W: torch.Tensor) -> dict:
    """Norms, interference and W^T W for one trained W (m, n)."""
    norms = W.norm(dim=0)                                     # (n,)
    unit = W / (norms + 1e-9)
    gram = W.T @ W                                            # (n, n)
    # Superposition of feature i: how much the other features project onto its direction.
    overlap = (unit.T @ W) ** 2                               # (n, n): (W_i_hat . W_j)^2
    interference = overlap.sum(1) - torch.diagonal(overlap)
    represented = norms > REPRESENTED
    return {
        "norms": [round(float(v), 3) for v in norms],
        "interference": [round(float(v), 3) for v in interference],
        "gram": [[round(float(v), 3) for v in row] for row in gram],
        "W": [[round(float(v), 3) for v in row] for row in W],
        "n_represented": int(represented.sum()),
        "n_represented_with_interference": int((represented & (interference > 0.1)).sum()),
    }


def run(figures: bool = False) -> dict:
    guard = Guard("toy.py")
    torch.set_num_threads(4)
    out: dict = {"model": "x' = ReLU(W^T W x + b), loss = sum_i I_i (x_i - x'_i)^2 (Elhage et al. 2022)",
                 "feature_distribution": "each feature 0 with probability S, else uniform on [0, 1]",
                 "sparsities": SPARSITIES, "seeds": list(SEEDS), "steps": STEPS, "batch": BATCH, "lr": LR,
                 "represented_rule": f"||W_i|| > {REPRESENTED}",
                 "interference_rule": "sum over j != i of (W_i_hat . W_j)^2; > 0.1 counts as sharing",
                 "setups": {}}
    for name, cfg in SETUPS.items():
        n, m = cfg["n"], cfg["m"]
        importance = torch.tensor([cfg["importance_decay"] ** i for i in range(n)])
        best_W, best_loss, best_seed = None, None, None
        all_losses = []
        for seed in SEEDS:
            W, b, loss = train(n, m, importance, SPARSITIES, seed)
            all_losses.append([round(float(v), 5) for v in loss])
            if best_W is None:
                best_W, best_loss, best_seed = W.clone(), loss.clone(), [seed] * len(SPARSITIES)
            else:
                better = loss < best_loss
                best_W[better] = W[better]
                best_loss[better] = loss[better]
                best_seed = [seed if bt else s for bt, s in zip(better.tolist(), best_seed)]
        frames = []
        for i, S in enumerate(SPARSITIES):
            d = describe(best_W[i])
            d.update({"sparsity": S, "loss": round(float(best_loss[i]), 5), "seed": best_seed[i]})
            frames.append(d)
        out["setups"][name] = {"n": n, "m": m, "importance": [round(float(v), 4) for v in importance],
                               "loss_by_seed": all_losses, "frames": frames}
        counts = [f["n_represented"] for f in frames]
        print(f"setup {name} (n={n}, m={m}): features represented by sparsity "
              + ", ".join(f"S={S}: {c}" for S, c in zip(SPARSITIES, counts)))
        guard.tick(f"setup {name}")

    if figures:
        fig_toy(out, FIGURES / "toy_superposition.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("toy", out)
    return out


def fig_toy(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B = res["setups"]["A"], res["setups"]["B"]
    S = res["sparsities"]
    lo, hi = 0, len(S) - 1
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.42, wspace=0.32)

    def gram_panel(ax, frame, title):
        g = np.array(frame["gram"])
        lim = max(1.0, np.abs(g).max())
        ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([0, 19]); ax.set_yticks([0, 19])
        ax.set_xlabel("feature j", fontsize=8.5); ax.set_ylabel("feature i", fontsize=8.5)

    def norm_panel(ax, frame, title):
        nrm = np.array(frame["norms"]); itf = np.array(frame["interference"])
        colors = ["#c1121f" if v > 0.1 else "#1f918d" for v in itf]
        ax.barh(range(len(nrm)), nrm, color=colors)
        ax.axvline(REPRESENTED, color="#888", lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.25)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("||W_i||  (teal: own direction, red: shared)", fontsize=8.5)
        ax.set_ylabel("feature (most important on top)", fontsize=8.5)

    fA0, fA1 = A["frames"][lo], A["frames"][hi]
    gram_panel(fig.add_subplot(gs[0, 0]), fA0, f"A: W$^T$W, S = {S[lo]}")
    norm_panel(fig.add_subplot(gs[0, 1]), fA0, f"A: {fA0['n_represented']} of 20 features kept, S = {S[lo]}")
    gram_panel(fig.add_subplot(gs[0, 2]), fA1, f"A: W$^T$W, S = {S[hi]}")
    norm_panel(fig.add_subplot(gs[0, 3]), fA1, f"A: {fA1['n_represented']} of 20 features kept, S = {S[hi]}")

    for col, idx in ((0, lo), (1, hi)):
        ax = fig.add_subplot(gs[1, col])
        W = np.array(B["frames"][idx]["W"])
        for i in range(W.shape[1]):
            ax.annotate("", xy=(W[0, i], W[1, i]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=plt.cm.viridis(i / 4), lw=1.8))
            ax.text(W[0, i] * 1.12, W[1, i] * 1.12, str(i), fontsize=8, ha="center", va="center")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
        ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
        ax.set_title(f"B: 5 features in 2 dims, S = {S[idx]}\n{B['frames'][idx]['n_represented']} kept",
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2:])
    xs = [1 - s for s in S]
    ax.plot(xs, [f["n_represented"] for f in A["frames"]], "o-", color="#b8336a",
            label="A: features kept (of 20, in 5 dims)")
    ax.plot(xs, [f["n_represented"] for f in B["frames"]], "s-", color="#453781",
            label="B: features kept (of 5, in 2 dims)")
    ax.axhline(5, color="#b8336a", lw=0.7, ls=":")
    ax.axhline(2, color="#453781", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("feature probability 1 − S (log scale; sparser to the right)", fontsize=8.5)
    ax.set_ylabel("features with ||W_i|| > 0.5", fontsize=8.5)
    ax.set_title("Dotted lines: number of dimensions", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Toy superposition: dense features get one dimension each; sparse ones share", fontsize=11)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    run(figures=args.figures)


if __name__ == "__main__":
    main()
