"""Load a small local instruct model on Apple MPS for the hand-rolled loop.

We use raw ``transformers`` here (no smolagents) so section 1 shows the whole
machine with nothing hidden: tokenizer -> generate -> text. The default is
SmolLM2-1.7B-Instruct because tool-calling needs a model with enough capability
to follow the ReAct format; a 360M fallback is available for slow machines but
it fails the format far more often (see the README's honest failure notes).

The model is loaded once and cached at module level so repeated agent runs in
the eval lab do not reload weights.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
FALLBACK_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

_CACHE: dict[str, tuple] = {}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(model_id: str = DEFAULT_MODEL):
    """Return (tokenizer, model, device), loading & caching on first call."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    _CACHE[model_id] = (tok, model, device)
    return _CACHE[model_id]


@torch.no_grad()
def generate(messages: list[dict], model_id: str = DEFAULT_MODEL,
             max_new_tokens: int = 256, stop: list[str] | None = None,
             temperature: float = 0.0) -> str:
    """Chat-format ``messages`` and decode a reply.

    ``temperature=0`` (the default) means greedy decoding: reproducible, which
    the section-1 demos and their success checks rely on. Pass ``temperature>0``
    to sample — the eval lab does this on purpose to surface run-to-run variance
    ("agents are stochastic systems, measure them").

    ``stop`` is a list of strings; generation is truncated at the first one that
    appears in the decoded continuation.
    """
    tok, model, device = load(model_id)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    sample = temperature and temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(sample),
        pad_token_id=tok.eos_token_id,
    )
    if sample:
        gen_kwargs.update(temperature=float(temperature), top_p=0.95)
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    out = model.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if stop:
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        text = text[:cut]
    return text.strip()
