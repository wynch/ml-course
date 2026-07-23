# The transformer inference loop, in Python and Zig side by side

This is the course's fourth [algorithm card](../../docs/algorithm-cards.md). The
algorithm is **autoregressive decoding**: run the trained tiny-GPT forward one
token at a time to generate text. The Python side is the module-04 model's
forward/generate (mirrored in
[`python/src/ref_model.py`](python/src/ref_model.py)); the Zig side is the
framework-free engine in [`zig/src/`](zig/src). They compute the *same logits*
— [`python/scripts/parity.py`](python/scripts/parity.py) proves it to a max
abs difference of ~3e-6.

What Zig reveals here: **buffer reuse across steps**, the **KV cache's exact
shape**, and where a decode loop actually spends its memory and time — all the
things `model.generate(...)` hides.

There are two phases, and telling them apart is the whole game:

- **Prefill** — push the prompt through once to warm up the KV cache.
- **Decode** — then emit one token per forward pass, each pass touching only the
  *new* token, reading cached keys/values for everything before it.

---

## 0. Embeddings: token + position

**Python** (`ref_model.py`)

```python
pos = torch.arange(T, device=idx.device)
x = self.wte(idx) + self.wpe(pos)          # (B, T, d_model)
```

**Zig** (`gpt.zig::step`, one token at absolute position `pos`)

```zig
const tok_row = self.w.wte[token * dm ..][0..dm];
const pos_row = self.w.wpe[pos * dm ..][0..dm];
for (0..dm) |i| s.x[i] = tok_row[i] + pos_row[i];
```

**Commentary.** PyTorch embeds a whole `(B, T)` batch of tokens at once; the Zig
engine embeds exactly one token into a single reused `d_model` vector `s.x` —
the *residual stream*. There is no batch and no time axis in flight: the decode
loop only ever holds the current token's activations, plus the cache. `wte` is
also the output head (tied weights), so this same `[vocab, d_model]` matrix
appears again at the very end.

---

## 1. One block, pre-norm: LayerNorm then attention then MLP

**Python**

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
```

**Zig** (`gpt.zig::step`, inner loop over layers)

```zig
layerNorm(s.xn, s.x, layer.ln1_w, layer.ln1_b, cfg.eps);
self.attention(layer, l, pos);            // result -> s.y
for (0..dm) |i| s.x[i] += s.y[i];         // residual

layerNorm(s.xn, s.x, layer.ln2_w, layer.ln2_b, cfg.eps);
linear(s.ff, s.xn, layer.fc_w, layer.fc_b, cfg.d_ff, dm);
for (s.ff) |*v| v.* = gelu(v.*);
linear(s.y, s.ff, layer.mproj_w, layer.mproj_b, dm, cfg.d_ff);
for (0..dm) |i| s.x[i] += s.y[i];         // residual
```

**Commentary.** `x + attn(ln1(x))` in Python allocates a fresh tensor for every
`+` and every sublayer output. In Zig the residual is an in-place `+=` into
`s.x`, and `s.xn`, `s.ff`, `s.y` are pre-allocated scratch buffers reused on
every layer of every token. The `eps = 1e-5` and the *biased* variance (divide
by N) inside `layerNorm` are the two details that, if you get them wrong, blow
the parity check — they are the usual suspects when a from-scratch transformer
"almost" matches its reference.

---

## 2. QKV projection and the KV cache update

**Python** (`MultiHeadAttention.forward`)

```python
q, k, v = self.qkv(x).split(self.d_model, dim=2)
q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
```

**Zig** (`gpt.zig::attention`)

```zig
linear(s.qkv, s.xn, layer.qkv_w, layer.qkv_b, 3 * dm, dm);
const q  = s.qkv[0 .. dm];
const k  = s.qkv[dm .. 2 * dm];
const vv = s.qkv[2 * dm .. 3 * dm];

// write THIS token's K and V into the cache at position `pos`
@memcpy(self.kcache[l][pos * dm ..][0..dm], k);
@memcpy(self.vcache[l][pos * dm ..][0..dm], vv);
```

**Commentary — this is the KV cache.** PyTorch computes Q, K, V for *all* `T`
positions every call and immediately throws K and V away when `forward` returns.
That is fine for training, ruinous for generation: token 100 would recompute the
keys and values of tokens 0–99 that never change. The Zig engine computes K and
V for the *one* new token and appends them to a persistent buffer:
`kcache[layer]` is `[block_size × d_model]`, and head `h` of position `p` lives
at `[p*d_model + h*d_head .. + d_head]`. Q, K, V for one token are just three
contiguous slices of the same `qkv` buffer — the `[3*d_model, d_model]` weight
packs them stacked, so no copy is needed to split them.

---

## 3. Causal attention over the cache

**Python** (all positions at once, future masked out)

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, H, T, T)
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y = att @ v                                                 # (B, H, T, d_head)
```

**Zig** (one query, against cached keys 0..pos — no mask needed)

```zig
for (0..cfg.n_head) |h| {
    const off = h * dh;
    const qh = q[off..][0..dh];
    for (0..pos + 1) |j| {                       // only positions 0..=pos exist
        const kh = self.kcache[l][j * dm + off ..][0..dh];
        var dot: f32 = 0;
        for (0..dh) |d| dot += qh[d] * kh[d];
        s.scores[j] = dot * scale;
    }
    softmax(s.scores[0 .. pos + 1]);
    const yh = s.y[off..][0..dh];
    for (0..dh) |d| yh[d] = 0;
    for (0..pos + 1) |j| {
        const vh = self.vcache[l][j * dm + off ..][0..dh];
        for (0..dh) |d| yh[d] += s.scores[j] * vh[d];
    }
}
```

**Commentary — the causal mask disappears.** Python builds a `T×T` score matrix
and then masks its upper triangle to `-inf` so a token can't see the future. In
the decode loop the future *literally does not exist yet* — the cache only holds
positions `0..pos` — so the mask is free: we just loop `j` up to `pos`. The
whole quadratic score matrix collapses to a single length-`(pos+1)` `scores`
vector, reused every step. This is exactly why the KV cache turns an O(T²)
per-step forward into an O(T) one; the
[timing figure](figures/kv_cache_timing.png) shows the resulting linear-vs-
quadratic curves, and the [cache schematic](figures/kv_cache_schematic.png)
draws what those buffers hold.

---

## 4. Final norm, tied head, and the decode driver

**Python** (`GPT.generate`, the loop frameworks hide)

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.cfg.block_size:]     # crop to context
    logits, _ = self(idx_cond)                   # FULL forward, every step
    logits = logits[:, -1, :] / temperature
    # (top-k) ...
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, 1)
    idx = torch.cat([idx, next_id], dim=1)
```

**Zig** (`main.zig::cmdGenerate`, KV-cache path)

```zig
var pos: usize = 0;
var logits: []f32 = undefined;
for (ids) |id| { logits = model.step(id, pos); pos += 1; }   // PREFILL
while (gen < n_new and pos < cfg.block_size) : (gen += 1) {   // DECODE
    const next = smp.sample(logits);
    try b.tok.decodeInto(&out, arena, next);
    logits = model.step(next, pos);
    pos += 1;
}
```

**Commentary — the loop frameworks hide, and the tied head.** After the last
block, `layerNorm(ln_f)` runs and then the logits are `x @ wte^T` — the token
embedding *reused* as the output projection (no separate matrix exists in the
weight file). Notice what Python's `generate` does: it calls `self(idx_cond)` —
the entire model — on the growing sequence every iteration. Without a cache
that's the O(n³) baseline; PyTorch's own `generate` relies on `use_cache=True`
under the hood to avoid it. The Zig version makes the two phases explicit:
`step` over the prompt is prefill, `step` per emitted token is decode. The
engine implements *both* strategies (`--no-cache` replays the whole sequence
each step) so you can measure the difference yourself — and because the
attention math is identical, `selfcheck` confirms the two paths produce bitwise-
equal logits.

---

## 5. Allocation strategy: an arena per generation, zero per-token allocs

The single most important systems idea in this module:

```zig
// main.zig — everything for one run lives on the process arena
const arena = init.arena.allocator();
// gpt.zig — Model.init carves ALL scratch + cache out of that arena, ONCE:
const s = Scratch{ .x = try arena.alloc(f32, dm), .xn = ..., .logits = ... };
kcache[l] = try arena.alloc(f32, cfg.block_size * dm);
```

**Commentary.** Generating 120 tokens performs **zero** heap allocations after
setup. The residual stream, the QKV buffer, the attention scores, the MLP
hidden, the logits, and both KV caches are allocated once in `Model.init` and
overwritten in place on every step. Contrast the Python loop, where every
`self(idx_cond)`, every `@`, every `softmax`, every `torch.cat` mints new
tensors that the allocator and garbage collector must chase. An arena is the
right tool because the lifetime is trivially simple: *everything* dies at once
when the program exits, so there is nothing to free individually — which is also
why `resetCache` is a no-op (the driver just controls how many cache positions
are "live" via the `pos` it passes).

This is the difference the card exists to show. Python tells you *what* decoding
computes. The Zig tells you *what the machine does*: a fixed set of buffers, a
cache that grows by one column per token, and a loop that — once you have
written it by hand — makes `model.generate(...)` stop being magic.
