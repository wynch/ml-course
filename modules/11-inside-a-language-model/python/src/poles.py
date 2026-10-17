"""Section 4: neuron poles in SmolLM2-360M, read on both signs.

In SmolLM2's SwiGLU MLP a neuron's value is

    value = silu(gate_proj(x)) * up_proj(x)        (one of 2,560 per layer)

and that is what ``down_proj`` reads. The value can be large and positive or
large and negative, so each neuron has two poles. Following the Goodfire
essay "Neural networks are already clean", we read a neuron by its top
contexts on BOTH poles: the 10 places in a 1M-token slice of wikitext-103
where it is most positive, and the 10 where it is most negative.

The three neurons are not hand-picked. The rule: in layers 4, 16 and 28
(early, middle, late), keep the neurons whose two poles are both alive (top
activation at least 1% of the layer's median top for that sign), then draw
one per layer with numpy seed 0.

This is a demo of the method on three neurons. The measurement (what share of
poles reads as one job) is a separate experiment, outside this course; this
script does not run it.

Run me:
    uv run python src/poles.py            # numbers, results/poles.json
    uv run python src/poles.py --figures  # also figures/neuron_poles.png
"""

from __future__ import annotations

import argparse
import gc

from common import (FIGURES, MODEL_ID, Guard, load_hf_model, load_tokenizer, wrap,
                    write_results)

import numpy as np
import torch
import torch.nn.functional as F

from corpus import N_TOKENS, WIKITEXT_FILE, WIKITEXT_ID, WIKITEXT_REVISION, fetch_text, token_chunks

LAYERS = (4, 16, 28)       # model.model.layers[i], counting from 0
SEQ_LEN = 256
BATCH = 16
CANDIDATES = 64            # running top-k kept per pole before de-duplication
TOP = 10                   # contexts shown per pole
BEFORE, AFTER = 20, 4      # context window around the firing token, in tokens
DEAD = 0.01                # alive = top >= 1% of the layer's median top
SEED = 0


class TopK:
    """Running top-k over the whole corpus, for every neuron of one layer at once.

    Keeps, per neuron, the K largest values seen so far, their flat token
    index, and the gate/up values at that token (to tell + x + from - x -).
    """

    def __init__(self, n_neurons: int, k: int, device):
        self.v = torch.full((k, n_neurons), -float("inf"), device=device)
        self.idx = torch.zeros((k, n_neurons), dtype=torch.long, device=device)
        self.gate = torch.zeros((k, n_neurons), device=device)
        self.up = torch.zeros((k, n_neurons), device=device)
        self.k = k

    def update(self, values, offset: int, gate, up):
        # values, gate, up: (tokens, n_neurons); offset = flat index of row 0
        k = min(self.k, values.shape[0])
        v, i = values.topk(k, dim=0)
        g = gate.gather(0, i)
        u = up.gather(0, i)
        allv = torch.cat([self.v, v])
        alli = torch.cat([self.idx, i + offset])
        allg = torch.cat([self.gate, g])
        allu = torch.cat([self.up, u])
        keep = allv.topk(self.k, dim=0).indices
        self.v, self.idx = allv.gather(0, keep), alli.gather(0, keep)
        self.gate, self.up = allg.gather(0, keep), allu.gather(0, keep)


@torch.no_grad()
def scan(model, tokenizer, chunks: torch.Tensor, guard: Guard | None = None) -> dict:
    """One pass over the corpus; returns {layer: {"pos": TopK, "neg": TopK}}."""
    lm = wrap(model, tokenizer)
    dev = model.device
    n = model.config.intermediate_size
    tops = {l: {"pos": TopK(n, CANDIDATES, dev), "neg": TopK(n, CANDIDATES, dev)} for l in LAYERS}
    for s in range(0, len(chunks), BATCH):
        x = chunks[s : s + BATCH].to(dev)
        with lm.trace(x) as tracer:
            parts = []
            for l in LAYERS:
                mlp = lm.model.layers[l].mlp
                gate = mlp.gate_proj.output          # before silu
                up = mlp.up_proj.output
                value = mlp.down_proj.input          # silu(gate) * up, what down_proj reads
                parts.append(torch.stack([gate, up, value]))
            saved = torch.stack(parts).save()        # (layers, 3, batch, seq, 2,560)
            tracer.stop()                            # nothing after the last layer we read
        offset = s * SEQ_LEN
        for l, t in zip(LAYERS, saved):
            gate, up, value = (a.reshape(-1, n).float() for a in t)
            tops[l]["pos"].update(value, offset, gate, up)
            tops[l]["neg"].update(-value, offset, gate, up)
        # tracer.stop() ends the forward with an exception; its traceback keeps
        # every frame of the forward (and its activations) alive until the
        # cycle collector runs. Without this, MPS memory grows ~1 GB per batch.
        del saved, t, gate, up, value
        gc.collect()
        if guard is not None and s % (BATCH * 40) == 0:
            guard.tick()
            print(f"  {s * SEQ_LEN:>9,} / {chunks.numel():,} tokens")
    return tops


def regime(gate: float, up: float) -> str:
    """Sign of silu(gate) and of up. silu(g) < 0 exactly when g < 0."""
    return ("+" if gate > 0 else "-") + "x" + ("+" if up > 0 else "-")


def pick_neurons(tops: dict, seed: int = SEED) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    picked, stats = {}, {}
    for l in LAYERS:
        top_pos = tops[l]["pos"].v[0].cpu().numpy()
        top_neg = tops[l]["neg"].v[0].cpu().numpy()        # magnitude of the most negative
        alive = (top_pos >= DEAD * np.median(top_pos)) & (top_neg >= DEAD * np.median(top_neg))
        candidates = np.flatnonzero(alive)
        picked[l] = int(rng.choice(candidates))
        stats[l] = {
            "neurons": int(len(top_pos)),
            "alive_both_poles": int(alive.sum()),
            "median_top_positive": round(float(np.median(top_pos)), 3),
            "median_top_negative": round(float(-np.median(top_neg)), 3),
        }
    return picked, stats


def contexts(tops: dict, layer: int, neuron: int, pole: str, flat_ids: torch.Tensor, tokenizer) -> dict:
    t = tops[layer][pole]
    sign = 1 if pole == "pos" else -1
    vals = t.v[:, neuron].cpu().numpy() * sign
    idx = t.idx[:, neuron].cpu().numpy()
    gate = t.gate[:, neuron].cpu().numpy()
    up = t.up[:, neuron].cpu().numpy()
    peak = float(vals[0])
    out, seen = [], set()
    for v, i, g, u in zip(vals, idx, gate, up):
        seq = int(i) // SEQ_LEN
        if seq in seen:                     # one context per 256-token row
            continue
        seen.add(seq)
        start = max(seq * SEQ_LEN, int(i) - BEFORE)
        stop = min((seq + 1) * SEQ_LEN, int(i) + AFTER + 1)
        toks = [tokenizer.decode([int(x)]) for x in flat_ids[start:stop]]
        out.append({
            "tokens": toks,
            "fire": int(i) - start,
            "activation": round(float(v), 3),
            "share_of_max": round(float(v) / peak, 3),
            "regime": regime(float(g), float(u)),
        })
        if len(out) == TOP:
            break
    return {"max": round(peak, 3), "contexts": out}


def run(figures: bool = False) -> dict:
    guard = Guard("poles.py")
    tok = load_tokenizer()
    text, fetch_info = fetch_text(tok)
    chunks = token_chunks(tok, text, N_TOKENS, SEQ_LEN)
    del text
    guard.tick("corpus")
    model = load_hf_model()
    guard.tick("load model")
    tops = scan(model, tok, chunks, guard)
    guard.tick("scan")

    picked, stats = pick_neurons(tops)
    flat = chunks.flatten()
    neurons = []
    for l in LAYERS:
        n = picked[l]
        entry = {"layer": l, "neuron": n, "id": f"L{l}/N{n}"}
        for pole in ("pos", "neg"):
            entry[pole] = contexts(tops, l, n, pole, flat, tok)
        neurons.append(entry)
        print(f"\nL{l}/N{n}  (+max {entry['pos']['max']}, -max {entry['neg']['max']})")
        for pole in ("pos", "neg"):
            for c in entry[pole]["contexts"][:4]:
                s = "".join(c["tokens"][: c["fire"]]) + "[[" + c["tokens"][c["fire"]] + "]]" + \
                    "".join(c["tokens"][c["fire"] + 1 :])
                print(f"  {pole} {c['activation']:>7.2f} {c['regime']}  {s!r}")
    guard.tick("contexts")

    out = {
        "model": MODEL_ID,
        "corpus": f"{WIKITEXT_ID}@{WIKITEXT_REVISION[:7]} {WIKITEXT_FILE}, first "
                  f"{chunks.numel():,} tokens ({len(chunks)} rows of {SEQ_LEN})",
        "corpus_fetch": fetch_info,
        "value": "silu(gate_proj(x)) * up_proj(x), the input of down_proj",
        "rule": f"layers {list(LAYERS)}; alive = both poles' top >= {DEAD:.0%} of the layer's "
                f"median top; one neuron per layer drawn with numpy default_rng({SEED})",
        "context_window": {"before": BEFORE, "after": AFTER, "one_per_row_of": SEQ_LEN},
        "layer_stats": {str(l): s for l, s in stats.items()},
        "neurons": neurons,
    }
    if figures:
        fig_poles(out, FIGURES / "neuron_poles.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("poles", out)
    return out


def _snippet(c: dict, before: int = 9, after: int = 2) -> tuple[str, str, str]:
    f = c["fire"]
    clean = lambda ts: "".join(ts).replace("\n", "⏎")
    return (clean(c["tokens"][max(0, f - before) : f]), clean([c["tokens"][f]]),
            clean(c["tokens"][f + 1 : f + 1 + after]))


def fig_poles(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neurons = res["neurons"]
    fig, axes = plt.subplots(len(neurons), 2, figsize=(15, 2.3 + 2.6 * len(neurons)))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.3)
    mono = dict(fontsize=8.5, fontfamily="monospace")
    # Layout is fixed above, so a width measured now holds at save time.
    # Width of one monospace character in axes units, measured once.
    probe = axes[0][0].text(0, 0, "M" * 40, **mono)
    fig.canvas.draw()
    bb = probe.get_window_extent().transformed(axes[0][0].transAxes.inverted())
    cw = bb.width / 40
    probe.remove()
    for row, nrn in zip(axes, neurons):
        for ax, pole, color in ((row[0], "pos", "#1f918d"), (row[1], "neg", "#c1121f")):
            cs = nrn[pole]["contexts"][:6]
            ax.set_xlim(0, 1)
            ax.set_ylim(len(cs), -0.9)
            ax.axis("off")
            sign = "+" if pole == "pos" else "−"
            ax.text(0, -0.6, f"{nrn['id']}  {sign} pole  (max {nrn[pole]['max']:+.2f})",
                    fontsize=10, weight="bold", color=color)
            for k, c in enumerate(cs):
                pre, hit, post = _snippet(c)
                ax.barh(k, c["share_of_max"] * 0.12, left=0, height=0.55, color=color, alpha=0.55)
                ax.text(0.135, k, f"{c['share_of_max']:.2f}", va="center", fontsize=7.5, color="#555")
                x0 = 0.19
                ax.text(x0, k, "…" + pre, va="center", **mono)
                x1 = x0 + (len(pre) + 1) * cw
                ax.text(x1, k, hit, va="center", weight="bold", color="white",
                        bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none"), **mono)
                ax.text(x1 + (len(hit) + 0.6) * cw, k, post + "…", va="center", **mono)
    fig.suptitle("Three SmolLM2-360M neurons (chosen by rule, seed 0), top contexts on both poles, "
                 "1M tokens of wikitext-103. Bar and number = activation as share of the pole's max.",
                 fontsize=10.5)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    run(figures=args.figures)


if __name__ == "__main__":
    main()
