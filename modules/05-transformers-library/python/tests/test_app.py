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

    # Build + launch the server headless, then immediately shut it down.
    demo.launch(prevent_thread_lock=True, show_error=True)
    try:
        assert demo.local_url is not None
    finally:
        demo.close()
