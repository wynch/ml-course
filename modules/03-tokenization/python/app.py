"""Gradio playground: type text, see it split into colored token spans.

A dropdown switches between three tokenizers:

  * your-BPE   — the from-scratch byte-level BPE (500 merges), loaded from the
                 committed merge list in ../artifacts/scratch_merges.txt
  * HF-trained — a byte-level BPE trained with the `tokenizers` library
  * SmolLM3    — the production 128k tokenizer (files only, no weights)

Run interactively:   uv run python app.py
The launch is also exercised headlessly in tests/test_app.py.
"""

from __future__ import annotations

import os
import sys

import gradio as gr

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "src"))

from bpe import BPETokenizer  # noqa: E402

MERGES_FILE = os.path.join(HERE, "..", "artifacts", "scratch_merges.txt")


def build_tokenizers() -> dict:
    """Construct the three tokenizers. Cheap: scratch + SmolLM3 just load."""
    scratch = BPETokenizer.from_merges(BPETokenizer.load_merges(MERGES_FILE))

    from hf_way import load_smollm3, train_hf_bpe

    hf = train_hf_bpe()
    smol = load_smollm3()
    return {"your-BPE": scratch, "HF-trained": hf, "SmolLM3": smol}


def _scratch_pieces(tok: BPETokenizer, text: str) -> list[str]:
    return [tok.vocab[i].decode("utf-8", errors="replace") for i in tok.encode(text)]


def tokenize(text: str, which: str, toks: dict) -> list[tuple[str, str]]:
    """Return HighlightedText tuples: (surface_string, token-index-as-label)."""
    if which == "your-BPE":
        pieces = _scratch_pieces(toks["your-BPE"], text)
    elif which == "HF-trained":
        pieces = [p.replace("Ġ", " ").replace("Ċ", "\n") for p in toks["HF-trained"].encode(text).tokens]
    else:
        pieces = [p.replace("Ġ", " ").replace("Ċ", "\n") for p in toks["SmolLM3"].tokenize(text)]
    # label each token by its position mod 8 so adjacent tokens get distinct colors
    return [(p, str(i % 8)) for i, p in enumerate(pieces)]


def build_demo() -> gr.Blocks:
    toks = build_tokenizers()

    def run(text: str, which: str):
        spans = tokenize(text, which, toks)
        return spans, f"{len(spans)} tokens"

    with gr.Blocks(title="Tokenization playground") as demo:
        gr.Markdown(
            "# Tokenization playground\n"
            "Type some text and watch it split into tokens. Switch tokenizers "
            "to see how a 500-merge toy compares to SmolLM3's 128k vocab."
        )
        with gr.Row():
            inp = gr.Textbox(
                label="text",
                value="The naïve café costs 4096 tokens. def f(x): return x+1",
                lines=3,
            )
            which = gr.Dropdown(
                choices=["your-BPE", "HF-trained", "SmolLM3"],
                value="SmolLM3",
                label="tokenizer",
            )
        count = gr.Markdown()
        out = gr.HighlightedText(label="tokens", combine_adjacent=False, show_legend=False)

        for comp in (inp, which):
            comp.change(run, [inp, which], [out, count])
        demo.load(run, [inp, which], [out, count])
    return demo


def main():
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
