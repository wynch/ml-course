"""Section 1: hooks, from scratch, on module 04's trained char-GPT.

A forward hook is a function PyTorch calls right after a module runs. It sees
the module's input and output. If it returns nothing, it only reads; if it
returns a tensor, that tensor replaces the output and the rest of the network
runs on the edited value. That is the whole mechanism behind every
interpretability library; this script uses nothing else.

What it does:
  1. records all 3 x 512 MLP neurons over 20,000 unseen characters and picks
     one by a stated rule: the neuron whose activation tracks "this character
     is a space" most closely (Pearson r);
  2. reads that neuron along one Shakespeare passage (the figure);
  3. edits it: a hook sets the neuron to zero, and we measure what the
     network loses, on the characters right after a space and elsewhere.

Run me:
    uv run python src/hooks.py            # print numbers, write results/hooks.json
    uv run python src/hooks.py --figures  # also draw figures/hook_neuron.png
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, write_results  # sets the MPS guard before torch

import numpy as np
import torch
import torch.nn.functional as F

from tiny_gpt import load_tiny_gpt, val_text

# Lines 35,715-35,717 of tiny_shakespeare.txt, inside the validation split (the
# last 10%), which module 04 never trained on.
PASSAGE = (
    "PETRUCHIO:\n"
    "Now, by the world, it is a lusty wench;\n"
    "I love her ten times more than e'er I did:"
)
N_CHARS = 20_000
BLOCK = 128


def record_neurons(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """GELU activations of every MLP neuron, per block: list of (T, 512).

    The hook sits on ``mlp.fc`` (the 128 -> 512 expansion). Its output is the
    pre-activation; the neuron's value is GELU of it, exactly what ``mlp.proj``
    reads next.
    """
    store: list[list[torch.Tensor]] = [[] for _ in model.blocks]

    def make_hook(i):
        def hook(module, inputs, output):
            store[i].append(F.gelu(output)[0].detach())  # returns None: read only
        return hook

    handles = [b.mlp.fc.register_forward_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    try:
        with torch.no_grad():
            # Non-overlapping chunks of BLOCK characters: the first characters
            # of each chunk see little context. That does not change which
            # neuron the module picks, but their activations are weaker.
            for s in range(0, len(ids), BLOCK):
                model(ids[s : s + BLOCK][None])
    finally:
        for h in handles:
            h.remove()  # a forgotten hook keeps firing on every later call
    return [torch.cat(chunks) for chunks in store]


def space_correlations(acts: list[torch.Tensor], text: str) -> torch.Tensor:
    """Pearson r of every neuron against 'this character is a space': (blocks, 512)."""
    is_space = torch.tensor([c == " " for c in text], dtype=torch.float32)
    f = (is_space - is_space.mean()) / is_space.std()
    rows = []
    for a in acts:
        z = (a - a.mean(0)) / (a.std(0) + 1e-8)
        rows.append((z * f[:, None]).mean(0))
    return torch.stack(rows)


def ablation_hook(neurons: list[int]):
    """Edit: zero the neurons. GELU(0) = 0, so we zero their pre-activations."""
    def hook(module, inputs, output):
        output = output.clone()
        output[..., neurons] = 0.0
        return output                         # returning a tensor replaces the output
    return hook


def ablate(model, plan: dict[int, list[int]]):
    """Register one zeroing hook per block; returns the handles to remove."""
    return [model.blocks[b].mlp.fc.register_forward_hook(ablation_hook(ns)) for b, ns in plan.items()]


def per_char_loss(model, ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy (nats) for predicting ids[t+1] from ids[:t+1], per position."""
    losses = []
    with torch.no_grad():
        for s in range(0, len(ids) - 1, BLOCK):
            x = ids[s : s + BLOCK]
            y = ids[s + 1 : s + 1 + BLOCK]
            x = x[: len(y)]
            logits, _ = model(x[None])
            losses.append(F.cross_entropy(logits[0], y, reduction="none"))
    return torch.cat(losses)


def run(figures: bool = False) -> dict:
    guard = Guard("hooks.py")
    model, tok = load_tiny_gpt("cpu")   # 0.6M parameters: CPU beats MPS launch overhead
    text = val_text()[:N_CHARS]
    ids = torch.tensor(tok.encode(text))
    guard.tick("load")

    acts = record_neurons(model, ids)
    r = space_correlations(acts, text)
    flat = int(r.flatten().argmax())
    b, n = divmod(flat, r.shape[1])
    pick = {"block": b, "neuron": n, "r": float(r[b, n]), "n_neurons_searched": r.numel()}
    top10 = r.flatten().topk(10).indices.tolist()
    plan10: dict[int, list[int]] = {}
    for f in top10:
        plan10.setdefault(f // r.shape[1], []).append(f % r.shape[1])
    a = acts[b][:, n]
    is_space = torch.tensor([c == " " for c in text])
    pick["mean_on_space"] = round(float(a[is_space].mean()), 3)
    pick["mean_elsewhere"] = round(float(a[~is_space].mean()), 3)
    pick["r"] = round(pick["r"], 3)
    guard.tick("record and pick")

    # Read along the passage.
    p_ids = torch.tensor(tok.encode(PASSAGE))
    line_acts = record_neurons(model, p_ids)[b][:, n]

    # Edit: rerun with the neuron zeroed.
    base = per_char_loss(model, ids)
    handles = ablate(model, {b: [n]})
    try:
        ablated = per_char_loss(model, ids)
        with torch.no_grad():
            line_ablated_logits, _ = model(p_ids[None])
    finally:
        for h in handles:
            h.remove()
    handles = ablate(model, plan10)
    try:
        ablated10 = per_char_loss(model, ids)
    finally:
        for h in handles:
            h.remove()
    with torch.no_grad():
        line_logits, _ = model(p_ids[None])
    after_space = is_space[: len(base)]           # predicting the first letter of a word
    edit = {
        "loss_all_before": round(float(base.mean()), 4),
        "loss_all_after": round(float(ablated.mean()), 4),
        "loss_after_space_before": round(float(base[after_space].mean()), 4),
        "loss_after_space_after": round(float(ablated[after_space].mean()), 4),
        "loss_elsewhere_before": round(float(base[~after_space].mean()), 4),
        "loss_elsewhere_after": round(float(ablated[~after_space].mean()), 4),
        "n_after_space": int(after_space.sum()),
        "n_elsewhere": int((~after_space).sum()),
        "top10_space_neurons": {str(k): v for k, v in plan10.items()},
        "top10_min_r": round(float(r.flatten().topk(10).values[-1]), 3),
        "loss_after_space_top10_zeroed": round(float(ablated10[after_space].mean()), 4),
        "loss_elsewhere_top10_zeroed": round(float(ablated10[~after_space].mean()), 4),
    }
    # One concrete case: what follows "it is a " in the passage, before and after.
    pos = PASSAGE.index("it is a ") + len("it is a ") - 1
    def top3(logits):
        p = F.softmax(logits[0, pos], -1)
        v, i = p.topk(3)
        return [[tok.decode([int(j)]), round(float(q), 3)] for q, j in zip(v, i)]
    edit["example_context"] = PASSAGE[: pos + 1].split("\n")[-1]
    edit["example_top3_before"] = top3(line_logits)
    edit["example_top3_after"] = top3(line_ablated_logits)
    guard.tick("ablate")

    out = {
        "model": "module 04 tiny char-GPT (artifacts/tiny_gpt_weights.bin), 3 blocks x 512 MLP neurons",
        "corpus": f"first {N_CHARS} characters of the validation split (last 10% of tiny_shakespeare.txt)",
        "rule": "the MLP neuron (post-GELU) with the highest Pearson r against 'this character is a space'",
        "neuron": pick,
        "passage": PASSAGE,
        "passage_activations": [round(float(v), 3) for v in line_acts],
        "edit": edit,
    }
    print(f"picked block {b} neuron {n}: r = {pick['r']} over {N_CHARS} chars "
          f"(mean {pick['mean_on_space']} on spaces, {pick['mean_elsewhere']} elsewhere)")
    print(f"zeroing it: loss after a space {edit['loss_after_space_before']} -> "
          f"{edit['loss_after_space_after']}, elsewhere {edit['loss_elsewhere_before']} -> "
          f"{edit['loss_elsewhere_after']}")
    print(f"zeroing the 10 most space-tracking neurons: after a space -> "
          f"{edit['loss_after_space_top10_zeroed']}, elsewhere -> {edit['loss_elsewhere_top10_zeroed']}")
    print(f"after {edit['example_context']!r}: {edit['example_top3_before']} -> {edit['example_top3_after']}")

    if figures:
        fig_hook(out, FIGURES / "hook_neuron.png")
    guard.tick("figures")
    out["guard"] = guard.report()
    write_results("hooks", out)
    return out


def fig_hook(res: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = res["passage"]
    acts = np.array(res["passage_activations"])
    b, n = res["neuron"]["block"], res["neuron"]["neuron"]
    e = res["edit"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), gridspec_kw={"height_ratios": [2.2, 1]})
    colors = ["#c1121f" if c == " " else "#1f918d" for c in text]
    ax.bar(range(len(text)), acts, color=colors, width=0.85)
    ax.set_xticks(range(len(text)))
    ax.set_xticklabels(["⏎" if c == "\n" else ("␣" if c == " " else c) for c in text],
                       fontfamily="monospace", fontsize=7.5)
    ax.set_xlim(-0.8, len(text) - 0.2)
    ax.set_ylabel("neuron value (after GELU)")
    ax.set_title(f"Block {b}, neuron {n}, read by a forward hook along unseen Shakespeare "
                 f"(red bars = spaces; r = {res['neuron']['r']} over 20,000 characters)", fontsize=10)
    ax.axhline(0, color="#888", lw=0.6)

    labels = ["first letter of a word\n(after a space)", "every other character"]
    series = [
        ("intact", "#9dada8", [e["loss_after_space_before"], e["loss_elsewhere_before"]]),
        (f"neuron {n} zeroed by a hook", "#e07a5f",
         [e["loss_after_space_after"], e["loss_elsewhere_after"]]),
        ("10 most space-tracking neurons zeroed", "#c1121f",
         [e["loss_after_space_top10_zeroed"], e["loss_elsewhere_top10_zeroed"]]),
    ]
    y = np.arange(2)
    for k, (label, color, vals) in enumerate(series):
        yy = y + (k - 1) * 0.27
        ax2.barh(yy, vals, height=0.25, color=color, label=label)
        for y1, v in zip(yy, vals):
            ax2.text(v + 0.02, y1, f"{v:.3f}", va="center", fontsize=7.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("next-character loss (nats), 20,000 unseen characters")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.invert_yaxis()
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
