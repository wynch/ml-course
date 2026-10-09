"""Byte-level Byte-Pair Encoding (BPE), built from scratch.

This is the simplest honest BPE that still teaches the whole idea:

  1. Start from the 256 raw bytes as the base vocabulary. Every possible input
     is representable, so there is never an "unknown token".
  2. Repeatedly find the most frequent adjacent pair of tokens in the corpus
     and merge it into a single new token. Each merge grows the vocabulary by
     one and (usually) shrinks the encoded sequence.
  3. The ordered list of merges *is* the model. Encoding replays the merges in
     order; decoding expands tokens back into their byte spans.

Deterministic tie-break (important!): when several pairs share the top count we
pick the lexicographically smallest pair ``(a, b)``. This makes training fully
reproducible and — crucially for this module — lets a completely separate Zig
implementation produce a byte-for-byte identical merge list. See ``../../zig``
and ``tests/test_cross_language.py``.

The whole file is deliberately un-clever: no incremental pair counts, no
regex pre-tokenization. We recount every pass, exactly like the Zig port, so
the two languages can be read side by side (see ``ALGORITHM.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


Pair = tuple[int, int]


def get_stats(ids: list[int]) -> dict[Pair, int]:
    """Count every adjacent pair of token ids in the sequence."""
    counts: dict[Pair, int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def best_pair(counts: dict[Pair, int]) -> Pair:
    """Pick the pair to merge: highest count, ties broken by smallest pair.

    Returning ``min`` over ``(-count, a, b)`` gives: maximum count first, then
    the lexicographically smallest ``(a, b)``. This exact rule is mirrored in
    the Zig trainer so both emit identical merges.
    """
    return min(counts, key=lambda p: (-counts[p], p[0], p[1]))


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``.

    Greedy, left-to-right, non-overlapping — the standard BPE merge.
    """
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """A trained byte-level BPE tokenizer.

    Attributes:
        merges: ordered list of merged pairs. Merge ``k`` produces token id
            ``256 + k``.
        vocab: token id -> the raw ``bytes`` it expands to.
        specials: name -> id for reserved special tokens (see exercise b).
    """

    merges: list[Pair] = field(default_factory=list)
    vocab: dict[int, bytes] = field(default_factory=dict)
    specials: dict[str, int] = field(default_factory=dict)

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, corpus: str | bytes, num_merges: int, verbose: bool = False) -> "BPETokenizer":
        """Learn ``num_merges`` merges from ``corpus`` (a str or raw bytes)."""
        data = corpus.encode("utf-8") if isinstance(corpus, str) else corpus
        ids = list(data)

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: list[Pair] = []

        for k in range(num_merges):
            counts = get_stats(ids)
            if not counts:
                break
            pair = best_pair(counts)
            new_id = 256 + k
            ids = merge(ids, pair, new_id)
            merges.append(pair)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {k:4d}: {pair} -> {new_id}  ({vocab[new_id]!r})")

        return cls(merges=merges, vocab=vocab)

    @classmethod
    def from_merges(cls, merges: list[Pair]) -> "BPETokenizer":
        """Rebuild a tokenizer from an ordered merge list (e.g. a prefix).

        A prefix of a trained merge list is itself a valid tokenizer — the one
        you would have gotten by training with fewer merges — so this is the
        cheap way to snapshot "the tokenizer at k merges" for the figures.
        """
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for k, (a, b) in enumerate(merges):
            vocab[256 + k] = vocab[a] + vocab[b]
        return cls(merges=list(merges), vocab=vocab)

    # ---- encode / decode ------------------------------------------------
    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode raw bytes by replaying the learned merges in order."""
        ids = list(data)
        for k, pair in enumerate(self.merges):
            ids = merge(ids, pair, 256 + k)
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a Python string (UTF-8) into token ids."""
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into a string.

        Special-token ids decode to their registered surface form; everything
        else expands through the byte vocab. ``errors="replace"`` keeps decode
        total even on deliberately malformed id streams.
        """
        id_to_special = {i: name for name, i in self.specials.items()}
        chunks: list[bytes] = []
        for i in ids:
            if i in id_to_special:
                chunks.append(id_to_special[i].encode("utf-8"))
            else:
                chunks.append(self.vocab[i])
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ---- convenience ----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials)

    def add_special(self, name: str) -> int:
        """Register a special token, returning its id (see exercise b)."""
        if name in self.specials:
            return self.specials[name]
        new_id = 256 + len(self.merges) + len(self.specials)
        self.specials[name] = new_id
        return new_id

    def save_merges(self, path: str) -> None:
        """Write merges as ``a b`` lines — the same format the Zig trainer emits."""
        with open(path, "w") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @staticmethod
    def load_merges(path: str) -> list[Pair]:
        merges: list[Pair] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                merges.append((int(a), int(b)))
        return merges


if __name__ == "__main__":
    import sys
    import time

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "../corpus/input.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    text = open(corpus_path, "rb").read()
    t0 = time.time()
    tok = BPETokenizer.train(text, n)
    dt = time.time() - t0
    print(f"trained {len(tok.merges)} merges on {len(text)} bytes in {dt:.2f}s")
    print(f"vocab size: {tok.vocab_size}")
    sample = "The quick brown fox."
    ids = tok.encode(sample)
    print(f"encode({sample!r}) -> {ids}")
    assert tok.decode(ids) == sample, "round-trip failed"
    print("round-trip OK")
