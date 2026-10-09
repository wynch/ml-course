"""Verify the reference solutions actually run against the real model.

These are slow (they load SmolLM2-360M once and run real forward passes), so
keep the token budgets small. The model is cached across tests in one process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

from common import build_chat_prompt, load_model_and_tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def model_tok():
    return load_model_and_tokenizer()


def test_greedy_is_deterministic_and_nonempty(model_tok):
    import solution_a_greedy as sol

    model, tok = model_tok
    prompt = build_chat_prompt(tok, "Name one primary color.")
    a = sol.greedy_decode(model, tok, prompt, max_new_tokens=12)
    b = sol.greedy_decode(model, tok, prompt, max_new_tokens=12)
    assert a == b  # greedy = argmax = deterministic
    assert len(a.strip()) > 0


def test_sampling_respects_seed(model_tok):
    import solution_a_greedy as sol

    model, tok = model_tok
    prompt = build_chat_prompt(tok, "Name one primary color.")
    a = sol.sample_decode(model, tok, prompt, max_new_tokens=12, seed=123)
    b = sol.sample_decode(model, tok, prompt, max_new_tokens=12, seed=123)
    assert a == b  # same seed -> same sample


def test_decision_layers_in_range(model_tok):
    import solution_b_decision_layer as sol

    model, tok = model_tok
    res = sol.find_decision_layers(model, tok, sol.PROMPTS)
    n_layers = model.config.num_hidden_layers
    assert set(res.keys()) == set(sol.PROMPTS)
    for prompt, layer in res.items():
        assert 0 <= layer <= n_layers


def test_tokens_per_sec_positive(model_tok):
    import solution_c_tokens_per_sec as sol

    model, tok = model_tok
    prompt = sol._make_prompt(tok, 8)
    tps = sol.time_generation(model, tok, prompt, new_tokens=8)
    assert tps > 0
