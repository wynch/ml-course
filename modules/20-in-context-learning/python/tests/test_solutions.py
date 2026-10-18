"""The reference solutions, checked against the scripts.

(a) runs in a second. (b) loads the two committed nets on the CPU. (c) needs
SmolLM2-360M-Instruct in the Hugging Face cache and is skipped without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "solutions"))

import common  # noqa: E402  (MPS guard before torch)
import icl  # noqa: E402


# ----------------------------------------------------------------- (a) rules
def test_rules_match_src():
    import solution_a_rules as sol

    rng = np.random.default_rng(3)
    for n in (2, 5, 9):
        X, y = rng.standard_normal((n, 5)), rng.standard_normal(n)
        assert np.allclose(sol.least_squares(X, y), icl.w_least_squares(X, y), atol=1e-8)
        assert np.allclose(sol.ridge(X, y), icl.w_ridge(X, y), atol=1e-10)
        assert np.allclose(sol.one_gd_step(X, y), icl.w_one_gd_step(X, y), atol=1e-12)


def test_rules_errors_match_results():
    import solution_a_rules as sol

    r = common.read_results("icl")["mse_over_d"]
    assert sol.mse_over_d(sol.least_squares, n=10) < 1e-20
    # 500 prompts against the module's 1,000: the same rule, within sampling noise
    assert abs(sol.mse_over_d(sol.one_gd_step, n=10) - r["one_gd_step"][10]) < 0.08
    assert abs(sol.mse_over_d(sol.one_gd_step, n=10, prompts=4000) - 6 / 16) < 0.03


# ----------------------------------------------------------------- (b) shuffled y
def test_shuffled_matches_src():
    import solution_b_shuffled as sol

    xs, ys, _ = icl.eval_set(4, shuffled=False, n=50)
    _, ys_s, _ = icl.eval_set(4, shuffled=True, n=50)
    assert np.array_equal(sol.shuffle_y(ys), ys_s)


def test_shuffled_scores():
    import solution_b_shuffled as sol

    xs, ys, _ = icl.eval_set(12, shuffled=False, n=500)
    ctrl = sol.score(sol.load("icl_control"), xs, sol.shuffle_y(ys))
    assert 0.8 <= ctrl.mean() <= 1.2
    assert sol.score(sol.load("icl_net"), xs, ys)[10] < 0.2


# ----------------------------------------------------------------- (c) label words
def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(common.MODEL_ID, "model.safetensors", revision=common.MODEL_REVISION), str)
    except Exception:
        return False


def test_label_prompt_matches_src():
    import fewshot
    import solution_c_label_words as sol

    demos = [("good fun .", 1), ("a bore .", 0)]
    for k in (0, 1, 2):
        assert sol.format_prompt("so so .", demos[:k]) == fewshot.format_prompt("so so .", demos[:k])


@pytest.mark.skipif(not _cached(), reason="SmolLM2-360M-Instruct not in the Hugging Face cache")
def test_label_margin_on_cpu():
    import fewshot
    import solution_c_label_words as sol

    tok = common.load_tokenizer()
    assert sol.label_ids(tok) == fewshot.label_token_ids(tok)
    model = common.load_hf_model("cpu")
    p = sol.format_prompt("an utterly charming and hilarious film", [("a dull , lifeless mess", 0)])
    assert sol.margin(model, tok, p) == pytest.approx(float(fewshot.score(model, tok, [p], fewshot.label_token_ids(tok))[0]), abs=1e-3)
