"""Exercise (c): spotting the same program twice.

An LLM often returns a program it was shown, with new comments or a new
docstring. Running it again would waste an evaluator call. Two programs
count as the same when their code is equal after parsing, with comments and
docstrings dropped.

Predict first: in evolve seed 0, what share of the 400 candidates is an exact
repeat by this rule?

Run the test: uv run pytest tests/test_solutions.py -k dedupe
"""

from __future__ import annotations

import ast


def normalize(code: str) -> str:
    """The code with comments, docstrings and spacing removed."""
    # TODO(you): ast.parse drops comments; drop a leading string in each
    # function body yourself, then ast.unparse.
    raise NotImplementedError


def count_repeats(codes: list[str]) -> int:
    """How many codes repeat an earlier one under normalize()."""
    # TODO(you)
    raise NotImplementedError


_ = ast
