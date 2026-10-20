"""Verify the Gradio app builds, its generate fn works, and it launches headless.

The launch uses ``prevent_thread_lock=True`` and is closed immediately, so no
server is ever left running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import app as app_module  # noqa: E402


def test_next_token_dist_shape():
    from common import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    df = app_module.next_token_dist(model, tok, prompt, temperature=0.7, k=10)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["token", "prob"]
    assert len(df) == 10
    assert df["prob"].iloc[0] >= df["prob"].iloc[-1]  # sorted descending


def test_generate_and_headless_launch():
    demo, generate = app_module.build_demo()

    history, dist, cleared = generate("Say hello in one word.", [], 0.0, 50, 0.9)
    assert cleared == ""
    assert history[-1]["role"] == "assistant"
    assert len(history[-1]["content"].strip()) > 0
    assert isinstance(dist, pd.DataFrame) and len(dist) > 0

    # The sliders redraw the panel: top-k = 3 leaves exactly three tokens with mass.
    narrow = demo.refresh_dist(history, 1.0, 3, 1.0)
    assert (narrow["prob"] > 0).sum() == 3
    assert abs(narrow["prob"].sum() - 1.0) < 1e-3
    assert demo.refresh_dist([], 1.0, 3, 1.0).empty

    # Build + launch the server headless, then immediately shut it down.
    demo.launch(prevent_thread_lock=True, show_error=True)
    try:
        assert demo.local_url is not None
    finally:
        demo.close()


def test_filtered_probs_applies_top_k_and_top_p():
    import torch

    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert (app_module.filtered_probs(logits, 1.0, top_k=2) > 0).sum() == 2
    kept = app_module.filtered_probs(logits, 1.0, top_p=0.75)
    assert (kept > 0).tolist() == [True, True, False, False]
    assert abs(float(kept.sum()) - 1.0) < 1e-6
    # As in generate(), top-p reads the top-k survivors renormalized: 0.5 / 0.8 = 0.625
    # already covers 0.6, so only one token is left.
    both = app_module.filtered_probs(logits, 1.0, top_k=2, top_p=0.6)
    assert (both > 0).tolist() == [True, False, False, False]


def test_filtered_probs_matches_generate_warpers():
    """The panel keeps exactly the tokens generate()'s own warpers keep."""
    import torch
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

    gen = torch.Generator().manual_seed(0)
    for temperature, top_k, top_p in [(0.7, 50, 0.9), (1.0, 5, 0.6), (1.5, 100, 0.3)]:
        logits = torch.randn(1000, generator=gen) * 3
        scores = logits[None].clone()
        for warp in (TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k),
                     TopPLogitsWarper(top_p)):
            scores = warp(None, scores)
        want = torch.isfinite(scores[0])
        got = app_module.filtered_probs(logits, temperature, top_k, top_p) > 0
        assert torch.equal(got, want)
