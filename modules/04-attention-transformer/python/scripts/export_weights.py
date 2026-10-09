"""Export the trained tiny-GPT to a language-agnostic binary blob.

Module 07 reimplements this model's *inference* in Zig. To make that possible
without reading any Python, we dump every weight tensor in a fixed, documented
order as raw little-endian float32, plus a JSON config describing dims and the
exact tensor order/shapes, plus the char vocab.

Outputs (all committed, under artifacts/):
  tiny_gpt_weights.bin     raw f32 tensors, concatenated in TENSOR_ORDER
  tiny_gpt_config.json     dims, counts, and the tensor manifest (name/shape/offset)
  tokenizer_chars.json     ordered char vocab (index == token id)

The byte layout is spelled out in artifacts/EXPORT_FORMAT.md.

Run:  uv run python scripts/export_weights.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.model import GPT, GPTConfig


def tensor_manifest(model: GPT, cfg: GPTConfig) -> list[tuple[str, torch.Tensor]]:
    """The canonical export order. A Zig reader consumes tensors in exactly
    this sequence. Linear weights are PyTorch layout `[out_features, in_features]`;
    y = x @ Wᵀ + b. LayerNorm has weight (gamma) then bias (beta).

    The LM head is TIED to `wte` — it is not emitted separately.
    """
    sd = model.state_dict()
    order: list[tuple[str, torch.Tensor]] = []
    order.append(("wte.weight", sd["wte.weight"]))          # [vocab, d_model]
    order.append(("wpe.weight", sd["wpe.weight"]))          # [block, d_model]
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        order += [
            (p + "ln1.weight", sd[p + "ln1.weight"]),       # [d_model]
            (p + "ln1.bias", sd[p + "ln1.bias"]),           # [d_model]
            (p + "attn.qkv.weight", sd[p + "attn.qkv.weight"]),  # [3*d_model, d_model]
            (p + "attn.qkv.bias", sd[p + "attn.qkv.bias"]),      # [3*d_model]
            (p + "attn.proj.weight", sd[p + "attn.proj.weight"]),# [d_model, d_model]
            (p + "attn.proj.bias", sd[p + "attn.proj.bias"]),    # [d_model]
            (p + "ln2.weight", sd[p + "ln2.weight"]),       # [d_model]
            (p + "ln2.bias", sd[p + "ln2.bias"]),           # [d_model]
            (p + "mlp.fc.weight", sd[p + "mlp.fc.weight"]), # [4*d_model, d_model]
            (p + "mlp.fc.bias", sd[p + "mlp.fc.bias"]),     # [4*d_model]
            (p + "mlp.proj.weight", sd[p + "mlp.proj.weight"]),  # [d_model, 4*d_model]
            (p + "mlp.proj.bias", sd[p + "mlp.proj.bias"]),      # [d_model]
        ]
    order.append(("ln_f.weight", sd["ln_f.weight"]))        # [d_model]
    order.append(("ln_f.bias", sd["ln_f.bias"]))            # [d_model]
    return order


def main() -> None:
    ckpt_path = C.MODELS / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    chars = ckpt["chars"]

    order = tensor_manifest(model, cfg)

    # --- write the raw f32 blob and build the manifest ---
    blob = bytearray()
    manifest = []
    offset = 0
    for name, t in order:
        arr = t.detach().cpu().contiguous().float().numpy().astype("<f4")
        raw = arr.tobytes()
        manifest.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "offset": offset,          # byte offset into the blob
                "count": int(arr.size),    # number of f32 elements
            }
        )
        blob += raw
        offset += len(raw)

    (C.ARTIFACTS / "tiny_gpt_weights.bin").write_bytes(bytes(blob))

    config = {
        "arch": "tiny-gpt-char",
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.d_model,
        "d_head": cfg.d_head,
        "d_ff": 4 * cfg.d_model,
        "layer_norm_eps": 1e-5,
        "tied_lm_head": True,
        "dtype": "float32",
        "byte_order": "little-endian",
        "linear_weight_layout": "[out_features, in_features] row-major; y = x @ W.T + b",
        "activation": "gelu",
        "n_tensors": len(manifest),
        "total_f32": offset // 4,
        "tensors": manifest,
    }
    (C.ARTIFACTS / "tiny_gpt_config.json").write_text(json.dumps(config, indent=2))
    (C.ARTIFACTS / "tokenizer_chars.json").write_text(
        json.dumps({"chars": chars}, ensure_ascii=False)
    )

    # --- self-check: reload the blob and compare against the model tensors ---
    raw = (C.ARTIFACTS / "tiny_gpt_weights.bin").read_bytes()
    for name, t in order:
        m = next(x for x in manifest if x["name"] == name)
        chunk = raw[m["offset"] : m["offset"] + m["count"] * 4]
        got = np.frombuffer(chunk, dtype="<f4").reshape(m["shape"])
        ref = t.detach().cpu().float().numpy()
        assert np.array_equal(got, ref), f"round-trip mismatch on {name}"

    total_bytes = len(raw)
    print(f"wrote {len(manifest)} tensors, {config['total_f32']:,} f32 values")
    print(f"tiny_gpt_weights.bin = {total_bytes:,} bytes ({total_bytes/1e6:.2f} MB)")
    print("round-trip self-check passed (blob == model tensors)")


if __name__ == "__main__":
    main()
