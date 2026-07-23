# tiny-GPT export format

This directory holds the trained tiny-GPT in a language-agnostic form so that
**module 07** can reimplement its *inference* in Zig without reading any Python.
Everything you need to load and run the forward pass is here.

## Files

| file | contents |
|------|----------|
| `tiny_gpt_weights.bin` | every weight tensor as raw little-endian `float32`, concatenated in a fixed order |
| `tiny_gpt_config.json` | model dims, hyper-params, and a per-tensor manifest (name, shape, byte offset, element count) |
| `tokenizer_chars.json` | `{"chars": [...]}` — the ordered char vocabulary; index == token id |

## Model architecture (recap)

A decoder-only transformer (GPT-style), char-level:

```
idx (T ints)
  → token_emb[idx] + pos_emb[0..T]            (both [*, d_model])
  → for each of n_layer blocks:
        x = x + attn(layernorm(x))            causal multi-head self-attention
        x = x + mlp (layernorm(x))            Linear→GELU→Linear, hidden = 4*d_model
  → layernorm_final(x)
  → logits = x @ wte.weightᵀ                  LM head is TIED to the token embedding
```

Config for the committed checkpoint: `vocab_size=65`, `block_size=128`,
`n_layer=3`, `n_head=4`, `d_model=128`, `d_head=32`, `d_ff=512`,
`layer_norm_eps=1e-5`, `activation=gelu`.

## Binary layout

`tiny_gpt_weights.bin` is a flat concatenation of tensors. There is **no
header** — the manifest in `tiny_gpt_config.json` (`"tensors"`) is the index.
Each manifest entry:

```json
{ "name": "...", "shape": [d0, d1, ...], "offset": <byte offset>, "count": <n f32> }
```

- `offset` is a byte offset from the start of the file.
- The tensor occupies `count * 4` bytes: `count` IEEE-754 `float32`s,
  **little-endian**, in **row-major (C) order**.
- `offset` of tensor k equals the sum of `count*4` of all earlier tensors, so you
  can also just read them sequentially in manifest order.

### Tensor order

Tensors appear in exactly this sequence (this is also `"tensors"` order):

1. `wte.weight` — token embedding, shape `[vocab_size, d_model]`
2. `wpe.weight` — position embedding, shape `[block_size, d_model]`
3. for each layer `i` in `0 .. n_layer-1`:
   1. `blocks.i.ln1.weight`  `[d_model]`   (LayerNorm gamma)
   2. `blocks.i.ln1.bias`    `[d_model]`   (LayerNorm beta)
   3. `blocks.i.attn.qkv.weight` `[3*d_model, d_model]`
   4. `blocks.i.attn.qkv.bias`   `[3*d_model]`
   5. `blocks.i.attn.proj.weight` `[d_model, d_model]`
   6. `blocks.i.attn.proj.bias`   `[d_model]`
   7. `blocks.i.ln2.weight`  `[d_model]`
   8. `blocks.i.ln2.bias`    `[d_model]`
   9. `blocks.i.mlp.fc.weight`   `[4*d_model, d_model]`
   10. `blocks.i.mlp.fc.bias`    `[4*d_model]`
   11. `blocks.i.mlp.proj.weight` `[d_model, 4*d_model]`
   12. `blocks.i.mlp.proj.bias`   `[d_model]`
4. `ln_f.weight` `[d_model]`  (final LayerNorm gamma)
5. `ln_f.bias`   `[d_model]`  (final LayerNorm beta)

The LM head is **not** stored: it is `wte.weight` reused, i.e.
`logits = x @ wte.weightᵀ`.

### Linear weight convention

All linear weights use PyTorch's `nn.Linear` layout: shape
`[out_features, in_features]`, and the layer computes

```
y = x @ W.T + b
```

So `W[o, i]` is the weight from input feature `i` to output feature `o`. In a
Zig reader that stores `W` row-major as `[out][in]`, computing
`y[o] = sum_i x[i] * W[o][i] + b[o]` is correct.

### QKV split

`attn.qkv.weight` `[3*d_model, d_model]` and `attn.qkv.bias` `[3*d_model]` pack
the query, key and value projections stacked along the output dimension in that
order: rows `[0 : d_model)` → Q, `[d_model : 2*d_model)` → K,
`[2*d_model : 3*d_model)` → V. Within Q (and K, V), the `n_head` heads are
contiguous slices of size `d_head` along that output dimension: head `h` uses
output rows `[h*d_head : (h+1)*d_head)`.

### Attention math (per head)

```
scores = (Q @ K.T) / sqrt(d_head)          # [T, T]
scores[i, j] = -inf  for j > i             # causal mask
weights = softmax(scores, axis=-1)         # rows sum to 1
head_out = weights @ V                      # [T, d_head]
```

Concatenate the `n_head` head outputs along the feature axis → `[T, d_model]`,
then apply `attn.proj`.

### LayerNorm

```
mean = mean(x)  ;  var = mean((x-mean)^2)          # over the d_model axis
y = (x - mean) / sqrt(var + eps) * gamma + beta    # eps = 1e-5
```

### GELU

The MLP uses GELU. Any standard GELU (exact erf form or the tanh
approximation) reproduces the samples to within sampling noise.

## Round-trip check

`scripts/export_weights.py` reloads the blob and asserts it is byte-identical to
the in-memory model tensors before finishing, so a file in this directory is
guaranteed consistent with its manifest.
