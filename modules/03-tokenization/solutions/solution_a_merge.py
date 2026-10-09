"""Solution (a) — the merge-application function."""

from __future__ import annotations


def merge_apply(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        # a hit needs a full pair, so guard against running off the end
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # skip both -> non-overlapping
        else:
            out.append(ids[i])
            i += 1
    return out


if __name__ == "__main__":
    assert merge_apply([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge_apply([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_apply([5, 6, 7], (0, 0), 9) == [5, 6, 7]
    print("ok")
