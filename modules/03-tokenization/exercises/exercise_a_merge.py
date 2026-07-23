"""Exercise (a) — implement the merge-application function.

This one function is the beating heart of both training and encoding: given a
sequence of token ids and a pair to merge, replace every occurrence of that
pair with a single new id.

Fill in `merge_apply` below, then check yourself:

    uv run pytest tests/test_solutions.py -k merge      # (against the solution)
    # or eyeball it:
    uv run python exercises/exercise_a_merge.py

The reference solution lives in ../solutions/solution_a_merge.py.
"""

from __future__ import annotations


def merge_apply(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``.

    Rules:
      * scan left to right
      * when ids[i], ids[i+1] == pair, emit new_id and skip BOTH elements
      * otherwise copy ids[i] through unchanged
      * "non-overlapping" means [1,1,1] with pair (1,1) -> [new_id, 1]
    """
    # TODO(you): build and return the merged list.
    # Hint: walk an index i with a while loop so you can advance by 2 on a hit.
    raise NotImplementedError("implement merge_apply")


if __name__ == "__main__":
    # Once implemented, these should all pass:
    assert merge_apply([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge_apply([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_apply([5, 6, 7], (0, 0), 9) == [5, 6, 7]
    print("merge_apply looks correct!")
