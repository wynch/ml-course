"""Reference solution — exercise (a), greedy decoding from scratch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import torch
import torch.nn.functional as F

from common import build_chat_prompt, load_model_and_tokenizer


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            last_logits = model(generated).logits[0, -1]  # (vocab,)

        next_id = last_logits.argmax().view(1, 1)  # (1, 1)
        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def sample_decode(model, tokenizer, prompt: str, max_new_tokens: int = 40,
                  temperature: float = 0.8, seed: int = 0) -> str:
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            last_logits = model(generated).logits[0, -1].float()

        probs = F.softmax(last_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).view(1, 1)

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
    a = greedy_decode(model, tok, prompt)
    b = greedy_decode(model, tok, prompt)
    print("greedy deterministic:", a == b)


if __name__ == "__main__":
    main()
