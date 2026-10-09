"""Shared helpers for module 05 — model id, device selection, loading.

Everything downloads to the *default Hugging Face cache* (``~/.cache/huggingface``),
never into the repo. The first call pulls ~700 MB; every call after is instant.
"""

from __future__ import annotations

import functools

import torch

# The workhorse for the whole module. ~700 MB, downloads to the HF cache on
# first use. The "same recipe, bigger" upgrade path is SmolLM3-3B — identical
# blocks, ~8x the parameters (see anatomy.py for the side-by-side table).
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
UPGRADE_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    On this course's Apple-silicon target that means **MPS** (the Metal GPU).
    Pass ``prefer="cpu"`` to force CPU — used by the tokens/sec exercise to
    compare the two lanes.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@functools.lru_cache(maxsize=4)
def load_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


@functools.lru_cache(maxsize=4)
def load_model(model_id: str = MODEL_ID, device: str | None = None):
    """Load the causal-LM in float32 on the chosen device.

    We keep float32 for numerical clarity in the logits/sampling labs — the
    model is small enough that fp32 on MPS is plenty fast. Cached so repeated
    calls in one process (tests, the Gradio app) reuse a single instance.
    """
    from transformers import AutoModelForCausalLM

    dev = get_device(device)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.to(dev)
    model.eval()
    return model


def load_model_and_tokenizer(model_id: str = MODEL_ID, device: str | None = None):
    return load_model(model_id, device), load_tokenizer(model_id)


def build_chat_prompt(tokenizer, user_message: str, system_message: str | None = None) -> str:
    """Apply the model's chat template and return the raw prompt *string*.

    Returning the string (not token ids) lets the sampling and chat-template
    labs show the exact text the model actually conditions on.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
