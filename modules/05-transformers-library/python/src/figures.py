"""Regenerate every figure in this module's README, in one shot.

    uv run python src/figures.py

Loads SmolLM2 once and hands it to each figure builder. ~1-2 min on MPS.
"""

from __future__ import annotations

from pathlib import Path

from common import load_model, load_tokenizer

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    model = load_model()
    tok = load_tokenizer()

    import anatomy
    import chat_template
    import embeddings
    import logit_lens
    import sampling_lab

    print("[1/7] param budget + comparison table")
    buckets = anatomy.print_architecture_summary(model, tok)
    anatomy.draw_param_figure(buckets, FIG_DIR / "param_budget.png")

    print("[2/7] next-token distribution")
    sampling_lab.fig_next_token_bar(model, tok, FIG_DIR / "next_token_dist.png")
    print("[3/7] temperature sweep")
    sampling_lab.fig_temperature_sweep(model, tok, FIG_DIR / "temperature_sweep.png")
    print("[4/7] top-k vs top-p")
    sampling_lab.fig_topk_topp(model, tok, FIG_DIR / "topk_topp.png")

    print("[5/7] logit lens")
    logit_lens.fig_logit_lens(model, tok, FIG_DIR / "logit_lens.png")

    print("[6/7] chat template")
    chat_template.fig_chat_template(tok, FIG_DIR / "chat_template.png")

    print("[7/7] embedding PCA")
    embeddings.fig_pca(model, tok, FIG_DIR / "embedding_pca.png")

    print("\nall figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
