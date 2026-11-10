"""Section 5: neurons or random directions?

If each MLP layer may pass on only k numbers per character, is it better to keep
k neurons or k random directions? Top-k keeps the k entries of the post-GELU
hidden vector h (512 per layer, the input of ``mlp.proj``) that are largest in
absolute value and zeroes the rest. In the random basis, h is first rotated by
a fixed random orthogonal Q, top-k runs on hQ, and the result is rotated back.

Two measures, both on module 04's validation split:

- **loss**, top-k at all three layers at once, every position (111,488 scored
  characters); the zero-MLP model sets every MLP output to zero.
- **output kept**, one layer at a time on clean activations of the first 128
  windows: 1 - sum ||P(h_k - h)||^2 / sum ||P h||^2, P the layer's proj weight.
  It needs no loss gap, so it also reads the untrained copy.

The **exchange rate** at k neurons is the number of random directions that do
as well, divided by k (straight line on log2 k between the powers of two that
bracket it). The **GELU null** keeps each neuron's mean and SD of its
pre-activation, draws independent Gaussians with them, applies GELU, and
measures output kept the same way: what GELU's shape and each neuron's scale
give with nothing else learned.

This is a one-model, two-rotation version of an earlier run (three retrained
seeds, five rotations). Rotations use that run's seeds for its first model.

Run me:
    uv run python src/basis.py            # numbers, results/basis.json
    uv run python src/basis.py --figures  # also figures/basis.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, write_results

import numpy as np
import torch
import torch.nn.functional as F

from sae import windows
from tiny_gpt import CORPUS, load_tiny_gpt, untrained_tiny_gpt

KS = [2**i for i in range(10)]     # 1 .. 512
WIDTH = 512
N_ROT = 2
CLEAN_WINDOWS = 128                # 16,384 characters for output kept and the null
K_READ = 8                         # the k the README reads
NULL_SEED = 55900


def rotation(layer: int, j: int) -> torch.Tensor:
    """The earlier run's rotation j for its first model: QR of a Gaussian, signs fixed by diag(R)."""
    rng = np.random.default_rng(55000 + 10 * layer + j)
    q, r = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
    return torch.tensor(q * np.sign(np.diag(r)), dtype=torch.float32)


def keep_topk(h: torch.Tensor, k: int, Q: torch.Tensor | None = None) -> torch.Tensor:
    """Keep the k entries of h largest in absolute value, in the neuron basis (Q None) or in hQ."""
    if k >= h.shape[-1]:
        return h
    z = h if Q is None else h @ Q
    idx = z.abs().topk(k, dim=-1).indices
    zk = torch.zeros_like(z).scatter(-1, idx, z.gather(-1, idx))
    return zk if Q is None else zk @ Q.T


def output_kept(h: torch.Tensor, P: torch.Tensor, k: int, Q: torch.Tensor | None = None) -> float:
    """1 - sum ||P(h_k - h)||^2 / sum ||P h||^2 over rows of h (tokens, 512); P is proj.weight."""
    err = (keep_topk(h, k, Q) - h) @ P.T
    return 1 - float(err.pow(2).sum() / (h @ P.T).pow(2).sum())


def k_equivalent(curve: list[float], value: float, increasing: bool) -> float:
    """The k at which `curve` (over KS) reaches `value`, on a straight line in log2 k."""
    c = np.asarray(curve, dtype=float)
    if not increasing:
        c, value = -c, -value
    if value <= c[0]:
        return 1.0
    if value >= c[-1]:
        return float(KS[-1])
    i = int(np.argmax(c >= value))
    f = (value - c[i - 1]) / (c[i] - c[i - 1])
    return float(2 ** (np.log2(KS[i - 1]) + f))


def exchange_rate(neuron: list[float], random: list[float], k: int, increasing: bool) -> float:
    return k_equivalent(random, neuron[KS.index(k)], increasing) / k


def frac3(neuron_loss: list[float], full: float) -> float:
    """Smallest share of the 512 neurons whose loss is within 3% of full, on a straight line in log2 k."""
    thr = 1.03 * full
    return k_equivalent(neuron_loss, thr, increasing=False) / WIDTH


@torch.no_grad()
def loss(model, x, y, edit=None, batch: int = 256) -> float:
    """Mean next-character loss (nats); ``edit(layer, h) -> h'`` changes proj's input in every block.
    edit == "zero" sets every MLP output to zero, proj's bias included."""
    hooks = []
    for i, b in enumerate(model.blocks):
        if edit == "zero":
            hooks.append(b.mlp.register_forward_hook(lambda m, inp, out: torch.zeros_like(out)))
        elif edit is not None:
            hooks.append(b.mlp.proj.register_forward_pre_hook(lambda m, inp, i=i: (edit(i, inp[0]),)))
    tot, n = 0.0, 0
    try:
        for s in range(0, len(x), batch):
            logits, _ = model(x[s : s + batch])
            t = y[s : s + batch]
            tot += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="sum"))
            n += t.numel()
    finally:
        for h in hooks:
            h.remove()
    return tot / n


@torch.no_grad()
def pre_activations(model, x) -> list[torch.Tensor]:
    """u = fc(ln2(x)) per layer, (tokens, 512): what GELU sees."""
    out, hooks = {}, []
    for i, b in enumerate(model.blocks):
        hooks.append(b.mlp.fc.register_forward_hook(
            lambda m, inp, o, i=i: out.setdefault(i, []).append(o.reshape(-1, o.shape[-1]))))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(out[i]) for i in range(len(model.blocks))]


def kept_curves(H: list[torch.Tensor], Ps: list[torch.Tensor], Qs) -> dict:
    """Output kept against k, mean over layers: neuron basis, and each rotation."""
    neu = [np.mean([output_kept(h, P, k) for h, P in zip(H, Ps)]) for k in KS]
    rnd = [[np.mean([output_kept(h, P, k, Qs[l][j]) for l, (h, P) in enumerate(zip(H, Ps))]) for k in KS]
           for j in range(N_ROT)]
    rmean = np.mean(rnd, 0).tolist()
    return {"neuron": [round(float(v), 5) for v in neu], "random_mean": [round(v, 5) for v in rmean],
            "rate_k8": round(exchange_rate(neu, rmean, K_READ, True), 3)}


def gelu_null(U: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for l, u in enumerate(U):
        g = torch.Generator().manual_seed(NULL_SEED + l)
        out.append(F.gelu(u.mean(0) + u.std(0) * torch.randn(u.shape, generator=g)))
    return out


def run(figures: bool = False) -> dict:
    guard = Guard("basis.py")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model, tok = load_tiny_gpt("cpu")
    untrained = untrained_tiny_gpt(1337, "cpu")
    text = CORPUS.read_text()
    split = len(text) - int(len(text) * 0.1)          # module 04's split
    xva, yva = windows(torch.tensor(tok.encode(text[split:])))
    Qs = [[rotation(l, j) for j in range(N_ROT)] for l in range(len(model.blocks))]

    full = loss(model, xva, yva)
    zero = loss(model, xva, yva, "zero")
    neuron = [loss(model, xva, yva, lambda l, h, k=k: keep_topk(h, k)) for k in KS[:-1]] + [full]
    random_each = [[loss(model, xva, yva, lambda l, h, k=k, j=j: keep_topk(h, k, Qs[l][j])) for k in KS[:-1]]
                   + [full] for j in range(N_ROT)]
    random = np.mean(random_each, 0).tolist()
    guard.tick("loss passes")

    clean = xva[:CLEAN_WINDOWS]
    U_tr, U_un = pre_activations(model, clean), pre_activations(untrained, clean)
    P_tr = [b.mlp.proj.weight for b in model.blocks]
    P_un = [b.mlp.proj.weight for b in untrained.blocks]
    with torch.no_grad():
        kept = {
            "trained": kept_curves([F.gelu(u) for u in U_tr], P_tr, Qs),
            "gelu_null": kept_curves(gelu_null(U_tr), P_tr, Qs),
            "untrained": kept_curves([F.gelu(u) for u in U_un], P_un, Qs),
        }
    guard.tick("output kept")

    i8 = KS.index(K_READ)
    gap = lambda v: (zero - v) / (zero - full)  # noqa: E731
    out = {
        "model": "module 04 tiny char-GPT (committed weights), top-k on the post-GELU input of mlp.proj",
        "untrained": "the same network under torch.manual_seed(1337) before step 1",
        "ks": KS,
        "rotations": N_ROT,
        "val_chars": int(xva.numel()),
        "clean_chars": int(clean.numel()),
        "loss_full": round(full, 4),
        "loss_zero_mlp": round(zero, 4),
        "loss_neuron": [round(v, 4) for v in neuron],
        "loss_random_each": [[round(v, 4) for v in r] for r in random_each],
        "loss_random_mean": [round(v, 4) for v in random],
        "gap_recovered_k8": {"neuron": round(gap(neuron[i8]), 3), "random": round(gap(random[i8]), 3)},
        "loss_rate_k8": round(exchange_rate(neuron, random, K_READ, False), 3),
        "loss_rate_k8_each_rotation": [round(exchange_rate(neuron, r, K_READ, False), 3) for r in random_each],
        "frac3": round(frac3(neuron, full), 4),
        "output_kept": kept,
        "trained_over_null_k8": round(kept["trained"]["rate_k8"] / kept["gelu_null"]["rate_k8"], 3),
        "mean_sd_of_u": {"trained": [round(float(u.std(0).mean()), 3) for u in U_tr],
                         "untrained": [round(float(u.std(0).mean()), 3) for u in U_un]},
    }
    print_table(out)
    if figures:
        fig_basis(out, FIGURES / "basis.png")
    out["guard"] = guard.report()
    write_results("basis", out)
    return out


def print_table(r: dict) -> None:
    print(f"held-out loss: full {r['loss_full']:.3f}, all MLPs zeroed {r['loss_zero_mlp']:.3f} nats")
    print(f"{'k':>4}  {'neurons':>8}  {'random':>8}")
    for k, n, m in zip(r["ks"], r["loss_neuron"], r["loss_random_mean"]):
        if k in (1, 8, 64, 256, 512):
            print(f"{k:>4}  {n:>8.3f}  {m:>8.3f}")
    rate = r["loss_rate_k8"]
    print(f"loss, k = 8: 8 neurons do what {8 * rate:.0f} random directions do ({rate:.2f}x; "
          f"rotations {' / '.join(f'{v:.2f}' for v in r['loss_rate_k8_each_rotation'])})")
    print(f"within 3% of full loss: {r['frac3']:.1%} of the neurons")
    ok = r["output_kept"]
    print("output kept, one layer at a time, k = 8, exchange rate:")
    for name in ("trained", "gelu_null", "untrained"):
        print(f"  {name:<10} {ok[name]['rate_k8']:.2f}x")
    print(f"  trained / GELU null: {r['trained_over_null_k8']:.2f}x")


def fig_basis(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NEU, RND, NULL = "#1f918d", "#b8336a", "#453781"
    ks = res["ks"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.plot(ks, res["loss_neuron"], "o-", color=NEU, label="k neurons")
    for j, r in enumerate(res["loss_random_each"]):
        ax.plot(ks, r, "o-", color=RND, alpha=0.8, label="k random directions" if j == 0 else None)
    ax.axhline(res["loss_zero_mlp"], color="#888", ls=":", lw=0.8)
    ax.axhline(1.03 * res["loss_full"], color="#888", ls="--", lw=0.8)
    ax.text(1.1, res["loss_zero_mlp"], "all MLPs zeroed", va="bottom", fontsize=8, color="#666")
    ax.text(1.1, 1.03 * res["loss_full"], "full + 3%", va="bottom", fontsize=8, color="#666")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k kept per layer and character (of 512)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("Trained model, top-k at all three layers", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ok = res["output_kept"]
    for name, ls in (("trained", "-"), ("untrained", "--")):
        ax.plot(ks, ok[name]["neuron"], ls, color=NEU, label=f"neurons, {name}")
        ax.plot(ks, ok[name]["random_mean"], ls, color=RND, label=f"random, {name}")
    ax.plot(ks, ok["gelu_null"]["neuron"], ":", color=NULL, lw=2, label="neurons, GELU null")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k kept (of 512)")
    ax.set_ylabel("MLP output kept (mean of 3 layers)")
    ax.set_title("One layer at a time, clean activations", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
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
