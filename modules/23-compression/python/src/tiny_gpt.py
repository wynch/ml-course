"""Load module 04's trained char-GPT from its committed export.

Module 04 commits the trained weights as one flat float32 file plus a manifest
(``artifacts/tiny_gpt_weights.bin`` and ``tiny_gpt_config.json``); its
``models/*.pt`` checkpoints are git-ignored. We rebuild the ``GPT`` class from
module 04's own ``model.py`` and fill it from that file, so this module hooks
exactly the network module 04 trained.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import torch

from common import M04

ARTIFACTS = M04 / "artifacts"
CORPUS = M04 / "corpus" / "tiny_shakespeare.txt"


def m04_model_module():
    # Loaded by path under its own name, so it never shadows another "model".
    spec = importlib.util.spec_from_file_location("m04_model", M04 / "python" / "src" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m04_model"] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.itos = dict(enumerate(chars))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_tiny_gpt(device: str = "cpu"):
    """Return (model, tokenizer) for module 04's trained tiny char-GPT."""
    cfg_json = json.loads((ARTIFACTS / "tiny_gpt_config.json").read_text())
    chars = json.loads((ARTIFACTS / "tokenizer_chars.json").read_text())["chars"]
    m04 = m04_model_module()
    cfg = m04.GPTConfig(
        vocab_size=cfg_json["vocab_size"],
        block_size=cfg_json["block_size"],
        n_layer=cfg_json["n_layer"],
        n_head=cfg_json["n_head"],
        d_model=cfg_json["d_model"],
        dropout=0.0,
    )
    model = m04.GPT(cfg)
    blob = np.fromfile(ARTIFACTS / "tiny_gpt_weights.bin", dtype="<f4")
    state = {}
    for t in cfg_json["tensors"]:
        start = t["offset"] // 4
        state[t["name"]] = torch.from_numpy(blob[start : start + t["count"]].reshape(t["shape"]).copy())
    # The head is tied to wte and is not in the file; the causal masks are buffers.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k == "head.weight" or k.endswith("attn.mask") for k in missing), missing
    return model.to(device).eval(), CharTokenizer(chars)


def val_text() -> str:
    """The last 10% of tiny-shakespeare: module 04's validation split, unseen in training."""
    text = CORPUS.read_text()
    return text[len(text) - int(len(text) * 0.1) :]


def new_gpt(d_model: int, n_layer: int = 3, n_head: int = 4, vocab_size: int = 65, block_size: int = 128):
    """A fresh, randomly initialised GPT of module 04's class (dropout off)."""
    m04 = m04_model_module()
    cfg = m04.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, d_model=d_model, dropout=0.0)
    return m04.GPT(cfg)


def splits(tok: CharTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Module 04's split: the first 90% of tiny-shakespeare trains, the last 10% validates."""
    text = CORPUS.read_text()
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(data) * 0.1)
    return data[:-n_val], data[-n_val:]
