"""Check the exercises against the real model.

By default the tests import the reference solutions, so they pass on a fresh
checkout. Set CHECK=exercises to import your files from exercises/ instead:

    CHECK=exercises uv run pytest tests/test_solutions.py -v

These are slow (they load SmolLM2-360M once and run real forward passes), so
keep the token budgets small. The model is cached across tests in one process.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
CHECK = os.environ.get("CHECK", "solutions")
if CHECK not in ("solutions", "exercises"):
    raise ValueError("CHECK must be 'solutions' or 'exercises'")
sys.path.insert(0, str(ROOT / CHECK))

from common import build_chat_prompt, load_model_and_tokenizer  # noqa: E402


def load(name: str):
    """Import e.g. solution_a_greedy, or exercise_a_greedy when CHECK=exercises."""
    prefix = "solution" if CHECK == "solutions" else "exercise"
    return importlib.import_module(f"{prefix}_{name}")


@pytest.fixture(scope="session")
def model_tok():
    return load_model_and_tokenizer()


def test_greedy_is_deterministic_and_nonempty(model_tok):
    sol = load("a_greedy")

    model, tok = model_tok
    prompt = build_chat_prompt(tok, "Name one primary color.")
    a = sol.greedy_decode(model, tok, prompt, max_new_tokens=12)
    b = sol.greedy_decode(model, tok, prompt, max_new_tokens=12)
    assert a == b  # greedy = argmax = deterministic
    assert len(a.strip()) > 0


def test_sampling_respects_seed(model_tok):
    sol = load("a_greedy")

    model, tok = model_tok
    prompt = build_chat_prompt(tok, "Name one primary color.")
    a = sol.sample_decode(model, tok, prompt, max_new_tokens=12, seed=123)
    b = sol.sample_decode(model, tok, prompt, max_new_tokens=12, seed=123)
    assert a == b  # same seed -> same sample


def test_decision_layers_in_range(model_tok):
    sol = load("b_decision_layer")

    model, tok = model_tok
    res = sol.find_decision_layers(model, tok, sol.PROMPTS)
    n_layers = model.config.num_hidden_layers
    assert set(res.keys()) == set(sol.PROMPTS)
    for prompt, layer in res.items():
        assert 0 <= layer <= n_layers


def test_tokens_per_sec_positive(model_tok):
    sol = load("c_tokens_per_sec")

    model, tok = model_tok
    prompt = sol._make_prompt(tok, 8)
    tps = sol.time_generation(model, tok, prompt, new_tokens=8)
    assert tps > 0


def test_decision_layers_match_the_readme(model_tok):
    sol = load("b_decision_layer")
    model, tok = model_tok
    res = sol.find_decision_layers(model, tok, sol.PROMPTS)
    assert [res[p] for p in sol.PROMPTS] == [27, 32, 26]


def test_lens_last_row_is_the_model_output(model_tok):
    """hidden_states[-1] is already normed; the lens must not norm it again."""
    import torch
    from logit_lens import PROMPT, layerwise_predictions

    model, tok = model_tok
    rows = layerwise_predictions(model, tok, PROMPT)["rows"]
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**ids).logits[0, -1].float(), dim=-1)
    assert abs(rows[-1]["top1_prob"] - float(probs.max())) < 1e-4
