"""Exercise (b) — find the layer where the model 'decides'.

Reuse the logit-lens machinery from ``src/logit_lens.py`` to locate, for each of
three prompts, the layer at which the model's top-1 prediction first becomes the
final answer and stays there. Different prompts "decide" at different depths.

Fill in the ``# TODO(you):`` and run:

    uv run python exercises/exercise_b_decision_layer.py

Check against the reference solution:

    uv run pytest tests/test_solutions.py -k decision -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from common import load_model_and_tokenizer
from logit_lens import decision_layer, layerwise_predictions

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The chemical symbol for gold is",
    "Two plus two equals",
]


def find_decision_layers(model, tokenizer, prompts: list[str]) -> dict[str, int]:
    """Return {prompt: decision_layer_index} for each prompt.

    Hint: call ``layerwise_predictions(model, tokenizer, prompt)`` to get a
    dict with a ``"rows"`` key, then pass those rows to ``decision_layer(...)``.
    """
    results: dict[str, int] = {}
    for prompt in prompts:
        # TODO(you): compute the per-layer predictions for `prompt`.
        res = ...  # replace

        # TODO(you): extract the decision layer from res["rows"].
        layer = ...  # replace

        results[prompt] = layer
    return results


def main() -> None:
    model, tok = load_model_and_tokenizer()
    results = find_decision_layers(model, tok, PROMPTS)
    n_layers = model.config.num_hidden_layers
    for prompt, layer in results.items():
        res = layerwise_predictions(model, tok, prompt)
        print(f'decides at layer {layer:>2}/{n_layers}  ->  {res["final_token"]!r:<10}  "{prompt}"')


if __name__ == "__main__":
    main()
