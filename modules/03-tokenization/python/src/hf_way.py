"""The Hugging Face way: the same idea, done with production tooling.

Two things happen here:

  1. We train a byte-level BPE tokenizer on the *same* corpus using the
     `tokenizers` library — the fast Rust implementation that ships inside
     `transformers`. A few lines replace our whole from-scratch trainer.
  2. We load SmolLM3's *production* tokenizer (tokenizer files only, no model
     weights) and put all three side by side on the same sentences to see how
     a real 128k-vocab tokenizer splits English, French, code, and numbers.

Run directly to print the comparison table:

    uv run python src/hf_way.py
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "input.txt")
SMOLLM3 = "HuggingFaceTB/SmolLM3-3B"

# A deliberately varied battery: English prose, French prose (accents!),
# Python code, and a number-heavy line. Fertility differs wildly across these.
SAMPLES = {
    "english": "The quick brown fox jumps over the lazy dog.",
    "french": "Le renard brun rapide saute par-dessus le chien paresseux à Noël.",
    "code": "def add(a, b):\n    return a + b  # sums two ints",
    "numbers": "Order 4096 units at $12.50 each on 2026-07-23.",
}


def train_hf_bpe(corpus_path: str = CORPUS, vocab_size: int = 512) -> Tokenizer:
    """Train a byte-level BPE tokenizer with the `tokenizers` library.

    This mirrors what GPT-2 / SmolLM3 do: byte-level pre-tokenization so every
    input is representable, then BPE merges up to ``vocab_size`` tokens.
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    )
    tok.train([corpus_path], trainer)
    return tok


def load_smollm3():
    """Load SmolLM3's production tokenizer (no torch, no model weights)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SMOLLM3)


def _clean(pieces: list[str]) -> list[str]:
    """Make byte-level artifacts (Ġ for space, Ċ for newline) human-readable."""
    return [p.replace("Ġ", "·").replace("Ċ", "\\n") for p in pieces]


def scratch_pieces(bpe, text: str) -> list[str]:
    """Surface strings for each token our from-scratch BPE emits."""
    out = []
    for i in bpe.encode(text):
        try:
            out.append(bpe.vocab[i].decode("utf-8"))
        except (UnicodeDecodeError, KeyError):
            out.append(bpe.vocab[i].decode("utf-8", errors="replace"))
    return out


def compare(bpe_scratch=None) -> dict:
    """Return a dict of {sample_name: {tokenizer: [pieces]}} for all three."""
    hf = train_hf_bpe()
    smol = load_smollm3()

    if bpe_scratch is None:
        import sys

        sys.path.insert(0, os.path.dirname(__file__))
        from bpe import BPETokenizer

        data = open(CORPUS, "rb").read()
        bpe_scratch = BPETokenizer.train(data, 500)

    result = {}
    for name, text in SAMPLES.items():
        result[name] = {
            "scratch-BPE (500 merges)": scratch_pieces(bpe_scratch, text),
            "HF-trained BPE (vocab 512)": _clean(hf.encode(text).tokens),
            "SmolLM3 (vocab 128k)": _clean(smol.tokenize(text)),
        }
    return result


def _fmt(pieces: list[str]) -> str:
    return " ".join(f"[{p}]" for p in pieces)


if __name__ == "__main__":
    table = compare()
    for name, by_tok in table.items():
        print(f"\n=== {name}: {SAMPLES[name]!r} ===")
        for tok_name, pieces in by_tok.items():
            print(f"  {tok_name:28s} ({len(pieces):3d} tok): {_fmt(pieces)}")
