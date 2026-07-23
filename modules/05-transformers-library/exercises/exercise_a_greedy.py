"""Exercise (a) — greedy decoding from scratch, no ``.generate``.

``model.generate`` hides the decode loop. Here you rebuild the simplest decoder
by hand: repeatedly run the model, take the *argmax* logit, append it, repeat.
Then contrast it with temperature sampling to feel why greedy is deterministic.

Fill in every ``# TODO(you):`` and run:

    uv run python exercises/exercise_a_greedy.py

Check against the reference solution:

    uv run pytest tests/test_solutions.py -k greedy -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import torch
import torch.nn.functional as F

from common import build_chat_prompt, load_model_and_tokenizer


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    """Greedily generate ``max_new_tokens`` tokens, one forward pass each.

    Return only the newly generated text (not the prompt). Stop early if the
    EOS token is produced.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # TODO(you): run the model on `generated` and grab the logits for
            # the LAST position only -> shape (vocab,).
            last_logits = ...  # replace

        # TODO(you): pick the most likely next token id (greedy = argmax).
        next_id = ...  # replace, shape (1, 1)

        generated = torch.cat([generated, next_id], dim=1)

        # TODO(you): stop if next_id is the EOS token.
        if ...:  # replace
            break

    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def sample_decode(model, tokenizer, prompt: str, max_new_tokens: int = 40,
                  temperature: float = 0.8, seed: int = 0) -> str:
    """Same loop, but SAMPLE from the temperature-scaled softmax instead of argmax."""
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            last_logits = model(generated).logits[0, -1].float()

        # TODO(you): turn `last_logits` into a probability distribution using
        # `temperature`, then sample ONE token id from it (torch.multinomial).
        probs = ...  # replace
        next_id = ...  # replace, shape (1, 1)

        generated = torch.cat([generated, next_id.to(generated.device)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    model, tok = load_model_and_tokenizer()
    prompt = build_chat_prompt(tok, "Name three primary colors.")
    print("greedy   :", greedy_decode(model, tok, prompt))
    print("sample   :", sample_decode(model, tok, prompt, temperature=0.8))
    # Greedy is deterministic: two runs must match.
    a = greedy_decode(model, tok, prompt)
    b = greedy_decode(model, tok, prompt)
    print("greedy deterministic:", a == b)


if __name__ == "__main__":
    main()
