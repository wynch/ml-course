"""Solution (c): spotting the same program twice."""

from __future__ import annotations

import ast


def normalize(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def count_repeats(codes: list[str]) -> int:
    seen: set[str] = set()
    repeats = 0
    for code in codes:
        key = normalize(code)
        if key in seen:
            repeats += 1
        seen.add(key)
    return repeats
