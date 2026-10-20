"""Section 2: a sparse autoencoder, from scratch, on module 04's char-GPT.

What it reads: the 512 MLP neurons of block 1 (post-GELU, the input of
``mlp.proj``), the same layer module 11 hooked. Training data: every
character of the training split (the first 90% of tiny-shakespeare);
evaluation: the validation split (the last 10%), which neither module 04 nor
this SAE trained on.

The SAE, TopK flavour (Gao et al. 2024):

    z    = TopK(ReLU(W_enc (x - b_pre) + b_enc))     2,048 latents, k kept
    x'   = W_dec z + b_pre                           decoder columns have norm 1

TopK rather than an L1 penalty: k IS the L0, so one knob sets sparsity
directly, and nothing pulls the kept values toward zero (L1 does, and the
reconstruction pays for it). We train one SAE per k in K_SWEEP and read
features from the k = K_MAIN one.

Measures, on the validation split: reconstruction MSE, fraction of variance
explained, L0, dead latents, and the next-character loss when the SAE's
reconstruction replaces the 512 neurons (then mlp.proj runs as usual).

Five latents are drawn by a stated rule (alive, numpy seed 0) and shown by
their top contexts, next to five raw neurons of the same layer drawn the same
way. A crude, automatic purity score compares all latents with all neurons.

Run me:
    uv run python src/sae.py            # numbers, results/sae.json
    uv run python src/sae.py --figures  # also figures/sae_*.png
"""

from __future__ import annotations

import argparse
import collections

from common import FIGURES, Guard, free, get_device, write_results

import numpy as np
import torch
import torch.nn.functional as F

from tiny_gpt import CORPUS, load_tiny_gpt

BLOCK = 1                  # module 11's layer
N_LATENTS = 2048           # 4x the 512 neurons
K_MAIN = 32
K_SWEEP = (8, 16, 32, 64)
STEPS = 2000
BATCH = 4096
LR = 1e-3
WINDOW = 128
ALIVE = 1e-3               # alive = fires on at least 0.1% of validation characters
N_SHOW = 5                 # features drawn for the README
N_BROWSE = 24              # latents exported to the explorable (the first five are the README's)
TOP = 10                   # contexts per feature
GAP = 24                   # firing positions at least this many characters apart
BEFORE, AFTER = 40, 6      # characters of context around the firing character
PURITY_TOP = 50
SEED = 0


class TopKSAE(torch.nn.Module):
    def __init__(self, d: int, n: int, k: int, b_pre: torch.Tensor):
        super().__init__()
        self.k = k
        w = torch.randn(d, n)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(w)                  # (d, n), unit columns
        self.W_enc = torch.nn.Parameter(w.T.clone())        # (n, d), tied at init
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_pre = torch.nn.Parameter(b_pre.clone())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x - self.b_pre) @ self.W_enc.T + self.b_enc)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)


def windows(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapping windows: inputs (N, 128) and next-character targets (N, 128)."""
    n = (len(ids) - 1) // WINDOW
    x = ids[: n * WINDOW].view(n, WINDOW)
    y = ids[1 : n * WINDOW + 1].view(n, WINDOW)
    return x, y


@torch.no_grad()
def collect(model, x: torch.Tensor, device, batch: int = 256) -> torch.Tensor:
    """Post-GELU block-1 neurons for every input character: (N*128, 512)."""
    out = []

    def hook(module, inputs):
        out.append(inputs[0].reshape(-1, inputs[0].shape[-1]))

    h = model.blocks[BLOCK].mlp.proj.register_forward_pre_hook(hook)
    try:
        for s in range(0, len(x), batch):
            model(x[s : s + batch].to(device))
    finally:
        h.remove()
    return torch.cat(out)


@torch.no_grad()
def lm_loss(model, x, y, device, replace=None, batch: int = 256) -> float:
    """Mean next-character loss (nats); ``replace(h) -> h'`` edits the 512 neurons first."""
    def pre(module, inputs):
        return (replace(inputs[0]),)

    h = model.blocks[BLOCK].mlp.proj.register_forward_pre_hook(pre) if replace else None
    tot, n = 0.0, 0
    try:
        for s in range(0, len(x), batch):
            logits, _ = model(x[s : s + batch].to(device))
            t = y[s : s + batch].to(device)
            tot += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="sum"))
            n += t.numel()
    finally:
        if h:
            h.remove()
    return tot / n


def train_sae(acts: torch.Tensor, k: int, seed: int = SEED, steps: int = STEPS) -> tuple[TopKSAE, list]:
    torch.manual_seed(seed)
    dev = acts.device
    sample = acts[torch.randperm(len(acts), device=dev)[:65536]]
    sae = TopKSAE(acts.shape[1], N_LATENTS, k, sample.mean(0)).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    var = ((sample - sample.mean(0)) ** 2).sum(-1).mean()
    curve = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(acts), (BATCH,), generator=gen).to(dev)
        x = acts[idx]
        xh, _ = sae(x)
        loss = ((x - xh) ** 2).sum(-1).mean() / var          # normalised MSE
        opt.zero_grad()
        loss.backward()
        # Drop the gradient part along each decoder column; renorm() below holds the
        # norm at 1 (Adam rescales coordinates, so the projection alone cannot).
        with torch.no_grad():
            g = sae.W_dec.grad
            g -= sae.W_dec * (g * sae.W_dec).sum(0, keepdim=True)
        opt.step()
        sae.renorm()
        if step % 100 == 0 or step == steps - 1:
            curve.append([step, round(float(loss.detach()), 5)])
    return sae.eval(), curve


@torch.no_grad()
def evaluate(sae: TopKSAE, acts: torch.Tensor, batch: int = 16384) -> dict:
    se, var_sum, l0, fires = 0.0, 0.0, 0.0, torch.zeros(N_LATENTS, device=acts.device)
    mean = acts.mean(0)
    for s in range(0, len(acts), batch):
        x = acts[s : s + batch]
        xh, z = sae(x)
        se += float(((x - xh) ** 2).sum())
        var_sum += float(((x - mean) ** 2).sum())
        nz = z > 0
        l0 += float(nz.sum())
        fires += nz.sum(0)
    n = len(acts)
    freq = (fires / n).cpu()
    return {
        "mse_per_char": round(se / n, 5),
        "fve": round(1 - se / var_sum, 4),
        "l0": round(l0 / n, 2),
        "dead": int((freq == 0).sum()),
        "alive_0.1pct": int((freq >= ALIVE).sum()),
        "freq": freq,
    }


def top_positions(values: torch.Tensor, top: int = TOP, gap: int = GAP) -> list[int]:
    """Indices of the largest values, each at least ``gap`` from the ones already kept."""
    order = torch.argsort(values, descending=True)[: top * 50].tolist()
    kept: list[int] = []
    for i in order:
        if values[i] <= 0:
            break
        if all(abs(i - j) >= gap for j in kept):
            kept.append(i)
        if len(kept) == top:
            break
    return kept


def contexts(values: torch.Tensor, text: str, top: int = TOP) -> dict:
    vmax = float(values.max())
    out = []
    for i in top_positions(values, top):
        a, b = max(0, i - BEFORE), min(len(text), i + AFTER + 1)
        out.append({"text": text[a:b], "fire": i - a, "activation": round(float(values[i]), 3),
                    "share_of_max": round(float(values[i]) / vmax, 3)})
    return {"max": round(vmax, 3), "freq": round(float((values > 0).float().mean()), 5), "contexts": out}


def purity(values: torch.Tensor, text: str, top: int = PURITY_TOP) -> float:
    """Share of a unit's top activations whose firing character is its most common one."""
    idx = torch.topk(values, top).indices.tolist()
    counts = collections.Counter(text[i] for i in idx)
    return counts.most_common(1)[0][1] / top


def run(figures: bool = False) -> dict:
    guard = Guard("sae.py")
    dev = get_device()
    model, tok = load_tiny_gpt(str(dev))
    text = CORPUS.read_text()
    split = len(text) - int(len(text) * 0.1)          # module 04's split
    train_ids = torch.tensor(tok.encode(text[:split]))
    val_text = text[split:]
    val_ids = torch.tensor(tok.encode(val_text))
    xtr, _ = windows(train_ids)
    xva, yva = windows(val_ids)
    val_chars = val_text[: xva.numel()]                # the characters the activations sit on
    acts_tr = collect(model, xtr, dev)
    acts_va = collect(model, xva, dev)
    guard.tick("collect")
    print(f"activations: train {tuple(acts_tr.shape)}, validation {tuple(acts_va.shape)}")

    mean_h = acts_tr.mean(0)
    base = {
        "intact": lm_loss(model, xva, yva, dev),
        "mean_ablated": lm_loss(model, xva, yva, dev, lambda h: mean_h.expand_as(h)),
        "zero_ablated": lm_loss(model, xva, yva, dev, lambda h: torch.zeros_like(h)),
    }
    print("next-char loss: " + ", ".join(f"{k} {v:.4f}" for k, v in base.items()))

    sweep, main = [], None
    for k in K_SWEEP:
        sae, curve = train_sae(acts_tr, k)
        ev = evaluate(sae, acts_va)

        def splice(h, sae=sae):
            shape = h.shape
            return sae(h.reshape(-1, shape[-1]))[0].reshape(shape)

        loss_sae = lm_loss(model, xva, yva, dev, splice)
        row = {k2: v for k2, v in ev.items() if k2 != "freq"}
        row.update({"k": k, "lm_loss_spliced": round(loss_sae, 4),
                    "loss_recovered_vs_mean_ablation": round(
                        (base["mean_ablated"] - loss_sae) / (base["mean_ablated"] - base["intact"]), 4),
                    "train_curve": curve})
        sweep.append(row)
        print(f"k={k}: FVE {row['fve']}, L0 {row['l0']}, dead {row['dead']}, "
              f"loss {loss_sae:.4f} (recovered {row['loss_recovered_vs_mean_ablation']})")
        if k == K_MAIN:
            main, main_freq = sae, ev["freq"]
        else:
            del sae
        guard.tick(f"sae k={k}")

    # Features of the main SAE, and neurons of the same layer, on the validation split.
    with torch.no_grad():
        z = torch.cat([main.encode(acts_va[s : s + 16384]) for s in range(0, len(acts_va), 16384)]).cpu()
    neurons = acts_va.cpu()
    rng = np.random.default_rng(SEED)
    alive = np.flatnonzero(main_freq.numpy() >= ALIVE)
    browse = [int(i) for i in rng.permutation(alive)[:N_BROWSE]]
    rng_n = np.random.default_rng(SEED)
    n_alive = np.flatnonzero((neurons > 0).float().mean(0).numpy() >= ALIVE)
    neuron_pick = [int(i) for i in rng_n.permutation(n_alive)[:N_SHOW]]
    latents = [{"id": f"L{i}", "index": i, **contexts(z[:, i], val_chars)} for i in browse]
    neuron_ctx = [{"id": f"N{i}", "index": i, **contexts(neurons[:, i], val_chars)} for i in neuron_pick]

    lat_pur = [purity(z[:, i], val_chars) for i in alive]
    neu_pur = [purity(neurons[:, i], val_chars) for i in range(neurons.shape[1])]
    pur = {
        "rule": f"share of a unit's top {PURITY_TOP} activations (validation) whose firing character is its most common one",
        "latents_alive": len(lat_pur),
        "neurons": len(neu_pur),
        "latents_median": round(float(np.median(lat_pur)), 3),
        "neurons_median": round(float(np.median(neu_pur)), 3),
        "latents_share_ge_0.9": round(float(np.mean(np.array(lat_pur) >= 0.9)), 3),
        "neurons_share_ge_0.9": round(float(np.mean(np.array(neu_pur) >= 0.9)), 3),
        "latents_hist": np.histogram(lat_pur, bins=10, range=(0, 1))[0].tolist(),
        "neurons_hist": np.histogram(neu_pur, bins=10, range=(0, 1))[0].tolist(),
    }
    print(f"purity: latents median {pur['latents_median']} ({pur['latents_share_ge_0.9']} >= 0.9), "
          f"neurons median {pur['neurons_median']} ({pur['neurons_share_ge_0.9']} >= 0.9)")
    freq_hist = np.histogram(np.log10(main_freq.numpy()[main_freq.numpy() > 0]), bins=24, range=(-6, 0))
    guard.tick("features")

    out = {
        "model": "module 04 tiny char-GPT, block 1 MLP neurons (post-GELU, input of mlp.proj), 512 wide",
        "sae": f"TopK SAE, {N_LATENTS} latents, decoder columns unit-norm, encoder tied at init; Adam lr {LR}, "
               f"{STEPS} steps of {BATCH} characters",
        "train_chars": int(acts_tr.shape[0]),
        "val_chars": int(acts_va.shape[0]),
        "k_main": K_MAIN,
        "lm_loss": {k: round(v, 4) for k, v in base.items()},
        "sweep": sweep,
        "freq_hist_log10": {"counts": freq_hist[0].tolist(), "edges": [round(float(e), 2) for e in freq_hist[1]]},
        "alive_rule": f"fires on at least {ALIVE:.1%} of validation characters",
        "pick_rule": f"alive latents of the k={K_MAIN} SAE in numpy.random.default_rng({SEED}).permutation order; "
                     f"the first {N_SHOW} go in the README, the first {N_BROWSE} in the explorable. Neurons: "
                     f"the same rule on the 512 neurons (alive = positive on at least 0.1% of characters)",
        "latents": latents,
        "neurons": neuron_ctx,
        "purity": pur,
    }
    free()
    if figures:
        fig_sae_metrics(out, FIGURES / "sae_metrics.png")
        fig_sae_features(out, FIGURES / "sae_features.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("sae", out)
    return out


def fig_sae_metrics(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sw = res["sweep"]
    ax = axes[0]
    ax.plot([r["l0"] for r in sw], [r["fve"] for r in sw], "o-", color="#b8336a")
    for r in sw:
        ax.annotate(f"k={r['k']}", (r["l0"], r["fve"]), textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L0: latents active per character")
    ax.set_ylabel("fraction of variance explained")
    ax.set_title("Reconstruction against sparsity", fontsize=10)

    ax = axes[1]
    lm = res["lm_loss"]
    labels = ["intact"] + [f"SAE k={r['k']}" for r in sw] + ["mean-ablated", "zero-ablated"]
    vals = [lm["intact"]] + [r["lm_loss_spliced"] for r in sw] + [lm["mean_ablated"], lm["zero_ablated"]]
    colors = ["#9dada8"] + ["#b8336a"] * len(sw) + ["#e07a5f", "#c1121f"]
    ax.barh(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 0.1, max(vals) + 0.25)
    ax.set_xlabel("next-character loss, validation (nats)")
    ax.set_title("Splice the reconstruction into block 1", fontsize=10)

    ax = axes[2]
    h = res["freq_hist_log10"]
    e = np.array(h["edges"])
    ax.bar(e[:-1], h["counts"], width=np.diff(e), align="edge", color="#453781")
    main = next(r for r in sw if r["k"] == res["k_main"])
    ax.axvline(np.log10(1e-3), color="#888", ls="--", lw=0.8)
    ax.set_xlabel("log10 firing frequency (validation)")
    ax.set_ylabel("latents")
    ax.set_title(f"k={res['k_main']}: {main['dead']} dead of {N_LATENTS}, "
                 f"{main['alive_0.1pct']} fire on ≥ 0.1%", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _show(c: dict) -> str:
    s = c["text"]
    f = c["fire"]
    vis = lambda t: t.replace("\n", "⏎")
    return vis(s[max(0, f - 28) : f]) + "[" + vis(s[f]) + "]" + vis(s[f + 1 : f + 5])


def fig_sae_features(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = [("latent", u) for u in res["latents"][:N_SHOW]] + [("neuron", u) for u in res["neurons"]]
    fig, axes = plt.subplots(2, N_SHOW, figsize=(17, 8.6))
    for ax, (kind, u) in zip(axes.flat, units):
        ax.axis("off")
        color = "#b8336a" if kind == "latent" else "#1f918d"
        ax.set_title(f"{kind} {u['index']}  (max {u['max']:.2f}, fires on {u['freq']:.1%})",
                     fontsize=9, color=color, loc="left")
        for j, c in enumerate(u["contexts"]):
            ax.text(0, 1 - j * 0.098, f"{c['share_of_max']:.2f} " + _show(c), fontsize=7.6,
                    fontfamily="monospace", va="top", transform=ax.transAxes)
    fig.suptitle("Top validation contexts, firing character in [brackets]. Top row: five SAE latents "
                 "(k = 32); bottom row: five raw neurons of the same layer. Both drawn by numpy seed 0.",
                 fontsize=10)
    fig.tight_layout()
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
