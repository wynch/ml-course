"""Exercise (c) — top-k sampling and the temperature knob.

At generation time the model gives you a probability distribution over the next
character. HOW you draw from it controls the character of the output:

  - temperature scales the logits before softmax. <1 sharpens (safer, repetitive),
    >1 flattens (wilder, more typos).
  - top-k keeps only the k most-likely tokens and renormalises, so the long tail
    of nonsense can never be sampled.

Implement `filter_logits` (temperature + optional top-k). The harness then loads
the trained model and samples the same prompt under several settings, writing a
comparison figure so you can see the effect.

Run:  uv run python exercises/ex_c_topk_sampling.py   (needs a trained model)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from src import config as C
from src.data import CharTokenizer
from src.model import GPT, GPTConfig


def filter_logits(
    logits: torch.Tensor,    # (vocab,) raw next-token logits
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """Return a probability distribution (sums to 1) over the vocabulary.

    Steps:
      1. divide logits by temperature (guard against 0)
      2. if top_k is set: keep only the top_k largest logits, set the rest to
         -inf so they get zero probability
      3. softmax to probabilities
    """
    # TODO(you): 1. scale by temperature
    logits = ...

    # TODO(you): 2. top-k filter (only if top_k is not None)
    #   hint: torch.topk(logits, top_k) gives the k largest values; the k-th
    #   largest is values[-1]. Set logits below that threshold to float("-inf").
    if top_k is not None:
        ...

    # TODO(you): 3. softmax -> probabilities
    probs = ...
    return probs


# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tok, prompt, n, temperature, top_k, device, seed=0):
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(n):
        cond = idx[:, -C.BLOCK_SIZE:]
        logits, _ = model(cond)
        probs = filter_logits(logits[0, -1], temperature, top_k)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return tok.decode(idx[0].tolist())


def main() -> None:
    import matplotlib.pyplot as plt

    device = C.get_device()
    ckpt = torch.load(C.MODELS / "ckpt_final.pt", map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.load(C.MODELS / "tokenizer_chars.json")

    settings = [
        ("T=0.5, top-k=none", 0.5, None),
        ("T=0.8, top-k=40", 0.8, 40),
        ("T=1.0, top-k=none", 1.0, None),
        ("T=1.5, top-k=10", 1.5, 10),
    ]
    prompt = "ROMEO:"
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 5.2))
    for ax, (label, temp, k) in zip(axes, settings):
        txt = generate(model, tok, prompt, 220, temp, k, device)
        ax.axis("off")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(0.0, 1.0, txt, family="monospace", fontsize=6.2, va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="#f4f1e8", ec="#bbb"))
        print(f"\n=== {label} ===\n{txt}")
    fig.suptitle("Sampling the trained tiny-GPT under different temperature / top-k",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = C.FIGURES / "ex_c_sampling_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
