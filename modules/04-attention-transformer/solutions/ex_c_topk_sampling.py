"""Solution (c) — top-k sampling and the temperature knob."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from src import config as C
from src.data import CharTokenizer
from src.model import GPT, GPTConfig


def filter_logits(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    # 1. temperature scaling (guard against divide-by-zero)
    logits = logits / max(temperature, 1e-8)

    # 2. top-k: keep only the k largest logits
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k).values[-1]     # k-th largest value
        logits = logits.masked_fill(logits < thresh, float("-inf"))

    # 3. softmax to probabilities
    probs = F.softmax(logits, dim=-1)
    return probs


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
