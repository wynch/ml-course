"""Exercise (c), solution: what a hybrid keeps between tokens, from its config alone.

Section 5 measured the cache of two models after 512, 4,096 and 16,384
tokens. Here you predict those byte counts from each model's config, before
looking at the measurement.

- An attention layer keeps a key and a value per token: 2 x kv_heads x head_dim
  numbers, in bfloat16 (2 bytes each).
- A Gated DeltaNet layer keeps one matrix per value head, key_dim x value_dim,
  in float32 (4 bytes), whatever the length; plus the last (kernel - 1) inputs
  of its short convolution over q, k and v (2 key widths + 1 value width), in
  bfloat16.

Predict before you run: Qwen3-0.6B has 28 attention layers of 8 kv heads of
128; Qwen3.5-0.8B has 6 attention layers of 2 kv heads of 256, and 18 Gated
DeltaNet layers of 16 heads of 128 x 128. Past how many tokens is Qwen3's cache
the larger? How large is each at 16,384 tokens?

    uv run python ../solutions/solution_c_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common


def cache_bytes(cfg: dict, layer_types: list[str] | None, T: int) -> int:
    """Bytes held after T tokens. layer_types None means every layer is attention."""
    n = cfg["num_hidden_layers"]
    types = layer_types or ["full_attention"] * n
    kv_per_token = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    total = 0
    for t in types:
        if t == "full_attention":
            total += kv_per_token * T
        else:
            hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
            dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
            state = hv * dk * dv * 4
            conv = (cfg["linear_conv_kernel_dim"] - 1) * (2 * hk * dk + hv * dv) * 2
            total += state + conv
    return total


def main() -> None:
    lib = common.read_results("library")
    for key, p in lib["models"].items():
        for T, row in p["contexts"].items():
            mine = cache_bytes(p["config"], p["layer_types"], int(T))
            print(f"{key:8s} T={int(T):6,d}: predicted {mine:>13,d} B, measured {row['cache_bytes']:>13,d} B")
    q3, q35 = lib["models"]["qwen3"], lib["models"]["qwen3.5"]
    T = 1
    while cache_bytes(q3["config"], None, T) <= cache_bytes(q35["config"], q35["layer_types"], T):
        T += 1
    print(f"Qwen3's cache passes Qwen3.5's at {T} tokens")


if __name__ == "__main__":
    main()
