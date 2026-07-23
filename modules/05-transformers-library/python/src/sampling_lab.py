"""Logits & sampling lab — the signature visuals of this module.

Everything here operates on the *raw logits* of a single forward pass: the
960-dim hidden state at the last position, projected through the tied output
head into 49,152 vocabulary scores. We then show the knobs that turn those
scores into a choice: temperature, top-k, top-p.

Run me:
    uv run python src/sampling_lab.py            # print top tokens for a prompt
    uv run python src/sampling_lab.py --figures  # draw all sampling figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import build_chat_prompt, load_model, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

# A confident prompt (one answer dominates) — good for the temperature sweep.
PROMPT = "The capital of France is"
# A deliberately open-ended prompt (flat distribution) — good for showing how
# top-p adapts: many plausible next tokens, so the nucleus is wide.
OPEN_PROMPT = "My favorite color is"


def next_token_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    """One forward pass → logits over the vocabulary for the next token."""
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
    return out.logits[0, -1].float().cpu()  # (vocab,)


def top_tokens(logits: torch.Tensor, tokenizer, k: int = 20):
    probs = F.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    toks = [tokenizer.decode([i]) for i in idx.tolist()]
    return toks, vals.tolist(), idx.tolist()


def _clean(tok: str) -> str:
    return tok.replace("\n", "\\n").replace(" ", "␣") or "∅"


# ---------------------------------------------------------------------------
def fig_next_token_bar(model, tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logits = next_token_logits(model, tokenizer, PROMPT)
    toks, probs, _ = top_tokens(logits, tokenizer, k=20)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(toks))
    ax.barh(list(y), probs, color="#4C72B0")
    ax.set_yticks(list(y))
    ax.set_yticklabels([_clean(t) for t in toks], fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("probability")
    ax.set_title(f'Next-token distribution (top 20)\nprompt: "{PROMPT}"', fontsize=11)
    for i, p in enumerate(probs):
        ax.text(p + 0.005, i, f"{p:.1%}", va="center", fontsize=8)
    ax.set_xlim(0, max(probs) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def fig_temperature_sweep(model, tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logits = next_token_logits(model, tokenizer, PROMPT)
    temps = [0.2, 0.7, 1.0, 1.5]
    # Anchor labels to the T=1.0 top tokens so bars line up across panels.
    _, _, base_idx = top_tokens(logits, tokenizer, k=12)
    labels = [_clean(tokenizer.decode([i])) for i in base_idx]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, T in zip(axes, temps):
        probs = F.softmax(logits / T, dim=-1)[base_idx].tolist()
        ax.bar(range(len(labels)), probs, color="#DD8452")
        ax.set_title(f"T = {T}", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontfamily="monospace", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probability")
    fig.suptitle(
        "Temperature reshapes the same logits: low = peaked/greedy, high = flat/creative",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def fig_topk_topp(model, tokenizer, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logits = next_token_logits(model, tokenizer, OPEN_PROMPT)
    probs = F.softmax(logits, dim=-1)
    vals, idx = probs.sort(descending=True)
    n = 25
    vals = vals[:n]
    labels = [_clean(tokenizer.decode([i])) for i in idx[:n].tolist()]

    k = 5
    p_thresh = 0.8
    cumsum = torch.cumsum(vals, dim=0)
    # nucleus: smallest set whose cumulative prob >= p_thresh
    nucleus_cut = int((cumsum < p_thresh).sum().item()) + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # top-k
    colors_k = ["#55A868" if i < k else "#cccccc" for i in range(n)]
    ax1.bar(range(n), vals.tolist(), color=colors_k)
    ax1.axvline(k - 0.5, color="#55A868", ls="--", lw=1.5)
    ax1.set_title(f"top-k (k={k}): keep the {k} most likely tokens, renormalize", fontsize=11)
    ax1.set_ylabel("probability")
    ax1.spines[["top", "right"]].set_visible(False)

    # top-p
    colors_p = ["#8172B3" if i < nucleus_cut else "#cccccc" for i in range(n)]
    ax2.bar(range(n), vals.tolist(), color=colors_p)
    ax2.axvline(nucleus_cut - 0.5, color="#8172B3", ls="--", lw=1.5)
    ax2.set_title(
        f"top-p (p={p_thresh}): keep the smallest set covering {p_thresh:.0%} of the mass "
        f"→ {nucleus_cut} tokens here",
        fontsize=11,
    )
    ax2.set_ylabel("probability")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontfamily="monospace", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f'top-k is a fixed count, top-p adapts to the distribution\nprompt: "{OPEN_PROMPT}"',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    model = load_model()
    tok = load_tokenizer()

    logits = next_token_logits(model, tok, PROMPT)
    toks, probs, _ = top_tokens(logits, tok, k=10)
    print(f'\nprompt: "{PROMPT}"\ntop-10 next tokens:')
    for t, p in zip(toks, probs):
        print(f"  {_clean(t):<12} {p:6.1%}")

    if args.figures:
        FIG_DIR.mkdir(exist_ok=True)
        fig_next_token_bar(model, tok, FIG_DIR / "next_token_dist.png")
        fig_temperature_sweep(model, tok, FIG_DIR / "temperature_sweep.png")
        fig_topk_topp(model, tok, FIG_DIR / "topk_topp.png")


if __name__ == "__main__":
    main()
