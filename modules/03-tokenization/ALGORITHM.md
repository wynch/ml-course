# The BPE training loop, in Python and Zig side by side

Both implementations run the *identical* algorithm with the *identical*
deterministic tie-break, which is why they emit byte-for-byte identical merge
lists (the cross-language test in `python/tests/test_cross_language.py` proves
it, and the full 500-merge lists match too). Reading them next to each other is
the fastest way to feel the difference between a high-level dynamic language and
a systems language doing exactly the same work.

- Python: [`python/src/bpe.py`](python/src/bpe.py)
- Zig: [`zig/src/main.zig`](zig/src/main.zig)

The loop has three moves, repeated `num_merges` times:

1. **count** every adjacent pair
2. **pick** the best pair (max count, ties → smallest pair)
3. **merge** that pair into a fresh id everywhere

---

## 0. Setup: the corpus as a list of token ids

We start from the raw bytes promoted to token ids in `[0, 256)`. Because every
byte is a token, nothing is ever "unknown".

**Python**

```python
data = corpus.encode("utf-8")      # bytes
ids = list(data)                   # list[int], each 0..255
vocab = {i: bytes([i]) for i in range(256)}
merges = []
```

**Zig**

```zig
const data = try cwd.readFileAlloc(io, corpus_path, alloc, .limited(64 * 1024 * 1024));
var ids = std.ArrayList(u32).empty;
try ids.ensureTotalCapacity(alloc, data.len);
for (data) |byte| ids.appendAssumeCapacity(@as(u32, byte));
```

**Commentary — memory layout.** Python's `list[int]` is an array of *pointers*
to heap-allocated `int` objects (though small ints 0–256 are interned, so those
particular pointers are shared). Every element is a machine word pointing
somewhere else — great for flexibility, terrible for cache locality. Zig's
`ArrayList(u32)` is a flat, contiguous `[]u32`: 4 bytes per token, packed, so a
scan streams straight through the CPU cache. We promote bytes to `u32` up front
because merged token ids will quickly exceed 255. `ensureTotalCapacity` +
`appendAssumeCapacity` pre-sizes the buffer so the initial fill does zero
reallocations.

---

## 1. Count adjacent pairs

**Python**

```python
def get_stats(ids):
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts
```

**Zig**

```zig
var counts = std.AutoHashMap(u64, u32).init(alloc);
defer counts.deinit();
var i: usize = 0;
while (i + 1 < ids.items.len) : (i += 1) {
    const key = packPair(ids.items[i], ids.items[i + 1]);
    const gop = try counts.getOrPut(key);
    if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
}
```

**Commentary — hash-map keys.** Python happily uses a `tuple[int, int]` as a
dict key: the tuple is hashable, but each one is a *heap-allocated object* built
fresh on every loop iteration, then hashed via Python's generic tuple hash. Zig
has no tuple-as-key convenience, so we pack the two ids into a single `u64`
(`a << 32 | b`) and key an `AutoHashMap(u64, u32)`. A `u64` key hashes in a
couple of instructions with no allocation. `getOrPut` returns a pointer to the
slot so we increment in place — no second lookup. This one design choice (packed
integer key vs. allocated tuple key) is a big part of why the Zig version is
~20× faster on the same corpus.

**Commentary — the `zip(ids, ids[1:])` trap.** In Python, `ids[1:]` *copies* the
whole tail list every call. On a 1M-element corpus that is a megabyte of
copying per merge. It is idiomatic and readable, and we keep it for teaching
clarity, but it is exactly the kind of hidden cost a systems language makes
visible: Zig just indexes `ids.items[i]` and `ids.items[i + 1]` with no copy.

---

## 2. Pick the best pair (the deterministic tie-break)

The rule: highest count wins; ties broken by the lexicographically smallest
pair `(a, b)`. This total order is what makes the two languages agree.

**Python**

```python
def best_pair(counts):
    return min(counts, key=lambda p: (-counts[p], p[0], p[1]))
```

**Zig**

```zig
var it = counts.iterator();
while (it.next()) |entry| {
    const a: u32 = @intCast(entry.key_ptr.* >> 32);
    const b: u32 = @intCast(entry.key_ptr.* & 0xffff_ffff);
    const count = entry.value_ptr.*;
    const better = !have_best or count > best_count or
        (count == best_count and (a < best_a or (a == best_a and b < best_b)));
    if (better) { best_count = count; best_a = a; best_b = b; have_best = true; }
}
```

**Commentary — order independence.** Both hash maps iterate in an *unspecified*
order, so we must never let "first seen" decide ties. Python's `min` with the
`(-count, a, b)` key and Zig's explicit `better` comparison encode the same
total order, so the winner is identical regardless of iteration order. Note we
unpack `a` and `b` back out of the `u64` key here — the packing is lossless, so
`>> 32` and `& 0xffffffff` recover the original ids.

---

## 3. Apply the merge

Greedy, left-to-right, non-overlapping: replace each occurrence of the pair with
the new id.

**Python**

```python
def merge(ids, pair, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
```

**Zig**

```zig
var out = std.ArrayList(u32).empty;
try out.ensureTotalCapacity(alloc, ids.items.len);
var j: usize = 0;
while (j < n) {
    if (j + 1 < n and ids.items[j] == best_a and ids.items[j + 1] == best_b) {
        out.appendAssumeCapacity(new_id); j += 2;
    } else {
        out.appendAssumeCapacity(ids.items[j]); j += 1;
    }
}
ids.deinit(alloc);
ids = out;
```

**Commentary — allocation strategy.** Both build a fresh output sequence rather
than mutating in place (mutating while shifting is fiddly and error-prone). In
Python, `out.append` may trigger periodic reallocation-and-copy as the list
grows; the interpreter hides it. In Zig we `ensureTotalCapacity(ids.len)` once —
the merged sequence is always ≤ the input length — so `appendAssumeCapacity`
never checks capacity or reallocates. We then `deinit` the old buffer and hand
ownership to the new one. This is the manual-memory tax: it is our job to free
the previous `ids` every iteration, whereas Python's garbage collector reclaims
the old list whenever it gets around to it.

---

## Why recount every pass?

Neither version keeps incremental pair counts between merges — we throw the
count map away and rebuild it each iteration. A production trainer (or the
`tokenizers` Rust library) maintains counts incrementally and only touches pairs
near a merge site, which is dramatically faster. We deliberately do *not*, for
two reasons: the naive loop is far easier to read, and — more importantly — it
is far easier to make two languages agree on the naive loop. Once you trust the
cross-language test, incremental counting is a natural next exercise.

## The scoreboard

Same algorithm, same corpus (`corpus/input.txt`, ~1.1 MB), same 500 merges:

| implementation | 500 merges | relative |
| --- | --- | --- |
| Python (`bpe.py`) | ~33 s | 1× |
| Zig (`main.zig`, ReleaseFast) | ~1.5 s | ~21× faster |

The algorithm is identical; the constant factor is the whole story.
