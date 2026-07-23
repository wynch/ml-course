"""Generate every figure embedded in the module README.

Figures written to ../../figures (kept small, <300KB each):

  1. compression_curve.png  — vocab size vs corpus compression as merges accrue
  2. token_length_hist.png  — token-length distribution at 100/300/500 merges
  3. colored_tokens.png      — the same texts split by 3 tokenizers, colored spans
  4. fertility.png           — tokens-per-word bar chart across languages

Run:  uv run python src/figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from bpe import BPETokenizer, best_pair, get_stats, merge  # noqa: E402

HERE = os.path.dirname(__file__)
CORPUS = os.path.join(HERE, "..", "..", "corpus", "input.txt")
FIGDIR = os.path.join(HERE, "..", "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

NUM_MERGES = 500
# A palette that reads clearly in both light and dark READMEs.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def train_with_curve(data: bytes, num_merges: int):
    """Train BPE while recording the encoded length after each merge.

    Returns (tokenizer, sizes) where sizes[k] is the number of tokens in the
    whole corpus after k merges (sizes[0] == len(data)).
    """
    ids = list(data)
    vocab = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    sizes = [len(ids)]
    for k in range(num_merges):
        counts = get_stats(ids)
        if not counts:
            break
        pair = best_pair(counts)
        ids = merge(ids, pair, 256 + k)
        merges.append(pair)
        vocab[256 + k] = vocab[pair[0]] + vocab[pair[1]]
        sizes.append(len(ids))
    return BPETokenizer(merges=merges, vocab=vocab), sizes


# --------------------------------------------------------------------------
def fig_compression(sizes: list[int]) -> str:
    original = sizes[0]
    merges_axis = list(range(len(sizes)))
    vocab_axis = [256 + k for k in merges_axis]
    compression = [original / s for s in sizes]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(vocab_axis, compression, color=PALETTE[0], lw=2)
    ax1.set_xlabel("vocabulary size (256 bytes + merges)")
    ax1.set_ylabel("compression ratio\n(bytes / tokens)", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(vocab_axis, [s / 1000 for s in sizes], color=PALETTE[1], lw=2, ls="--")
    ax2.set_ylabel("corpus length (K tokens)", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])

    ax1.set_title(f"Diminishing returns: {len(sizes) - 1} merges on tiny-shakespeare")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "compression_curve.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_token_length_hist(tok500: BPETokenizer, data: bytes) -> str:
    # Encode a sample slice (fast) at three merge-count snapshots.
    sample = data[:60_000]
    checkpoints = [100, 300, 500]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, color in zip(checkpoints, PALETTE):
        tok = BPETokenizer.from_merges(tok500.merges[:c])
        lengths = [len(tok.vocab[i]) for i in tok.encode_bytes(sample)]
        ax.hist(
            lengths,
            bins=range(1, 12),
            alpha=0.55,
            label=f"{c} merges (vocab {256 + c})",
            color=color,
            edgecolor="white",
        )
    ax.set_xlabel("token length (bytes)")
    ax.set_ylabel("count (60KB sample)")
    ax.set_title("Tokens get longer as merges accumulate")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGDIR, "token_length_hist.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_colored_tokens(table: dict) -> str:
    """Draw each sample as rows of colored token spans, one row per tokenizer."""
    names = list(table.keys())
    tok_names = list(next(iter(table.values())).keys())
    # Short tags keep the left gutter clear of the token boxes.
    short = {tok_names[0]: "scratch", tok_names[1]: "HF-BPE", tok_names[2]: "SmolLM3"}
    n_rows = len(names) * len(tok_names)

    start_x = 22.0
    char_w = 0.95
    xlim = 150.0
    fig, ax = plt.subplots(figsize=(13, 0.5 * n_rows + 1))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    y = n_rows
    for sample_name in names:
        group_top = y
        for tok_name in tok_names:
            pieces = table[sample_name][tok_name]
            x = start_x
            ax.text(9, y + 0.5, short[tok_name], fontsize=8, va="center", ha="left",
                    family="monospace", color="#888")
            for pi, piece in enumerate(pieces):
                disp = piece if piece.strip() else piece.replace(" ", "·")
                w = max(len(disp), 1) * char_w + 0.5
                ax.add_patch(Rectangle((x, y + 0.12), w, 0.76,
                             facecolor=PALETTE[pi % len(PALETTE)],
                             alpha=0.78, edgecolor="white", lw=0.7))
                ax.text(x + w / 2, y + 0.5, disp, fontsize=6, va="center",
                        ha="center", family="monospace", color="white")
                x += w + 0.35
            y -= 1
        # sample name label, vertically centered on the group, far left
        ax.text(1, group_top - 1.0, sample_name, fontsize=8.5, va="center",
                ha="left", color="#222", style="italic", rotation=90)
        y -= 0.4

    ax.set_title("Same text, different tokenizers — each box is one token", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "colored_tokens.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_fertility(tok_scratch: BPETokenizer) -> str:
    """Bar chart: tokens-per-word for English / French / code, 3 tokenizers.

    Fertility = tokens / whitespace-word. Higher = the tokenizer is "less fair"
    to that language: it spends more tokens to say the same thing.
    """
    from hf_way import SAMPLES, load_smollm3, train_hf_bpe

    langs = ["english", "french", "code"]
    hf = train_hf_bpe()
    smol = load_smollm3()

    def words(t):
        return max(len(t.split()), 1)

    series = {
        "scratch-BPE": [len(tok_scratch.encode(SAMPLES[l])) / words(SAMPLES[l]) for l in langs],
        "HF-trained": [len(hf.encode(SAMPLES[l]).tokens) / words(SAMPLES[l]) for l in langs],
        "SmolLM3": [len(smol.tokenize(SAMPLES[l])) / words(SAMPLES[l]) for l in langs],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(langs))
    width = 0.26
    for i, (name, vals) in enumerate(series.items()):
        ax.bar([xi + (i - 1) * width for xi in x], vals, width,
               label=name, color=PALETTE[i], edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(langs)
    ax.set_ylabel("tokens per word (fertility)")
    ax.set_title("Tokenizers are unfair across languages\n(lower is more efficient)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "fertility.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    data = open(CORPUS, "rb").read()
    print(f"training {NUM_MERGES} merges on {len(data)} bytes (this takes ~40s)...")
    tok, sizes = train_with_curve(data, NUM_MERGES)
    print("  done. generating figures...")

    paths = []
    paths.append(fig_compression(sizes))
    paths.append(fig_token_length_hist(tok, data))

    from hf_way import compare

    table = compare(bpe_scratch=tok)
    paths.append(fig_colored_tokens(table))
    paths.append(fig_fertility(tok))

    for p in paths:
        kb = os.path.getsize(p) / 1024
        print(f"  wrote {os.path.basename(p):24s} {kb:6.1f} KB")


if __name__ == "__main__":
    main()
