"""Reference solution — exercise (b), the decision layer via logit lens."""

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
    results: dict[str, int] = {}
    for prompt in prompts:
        res = layerwise_predictions(model, tokenizer, prompt)
        results[prompt] = decision_layer(res["rows"])
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
