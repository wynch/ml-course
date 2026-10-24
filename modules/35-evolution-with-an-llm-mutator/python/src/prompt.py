"""The prompt the mutator sees, and how its reply becomes a candidate program."""

from __future__ import annotations

import ast
import re

HEADER = """You are improving a heuristic for online bin packing. Items arrive one at a time and \
each must be placed at once into an open bin with enough room; a new bin is opened only when \
no open bin can take the item. For each item, `priority(item, bins)` receives the item size \
(a float) and a numpy array `bins` holding the remaining room of every open bin that can take \
the item. It returns one score per bin, and the item goes into the bin with the highest score. \
The goal is to use as few bins as possible.

{shown_line} Write \
priority_v{k}, a new version that you expect to use fewer bins than every version above. Use only \
numpy (as np) and math. Return a numpy array with one score per bin, never NaN. Keep it \
short, under 25 lines, with few comments. Reply with one Python code block that holds only \
the function priority_v{k}."""


def rename(code: str, name: str) -> str:
    return re.sub(r"\bdef\s+priority\w*\s*\(", f"def {name}(", code, count=1)


def build(parents: list[str]) -> str:
    """parents: program codes, worst first. Each is shown as priority_v0, v1, ..."""
    k = len(parents)
    shown = "\n\n".join(rename(code, f"priority_v{i}").rstrip() for i, code in enumerate(parents))
    line = ("Below is one version of the function." if k == 1 else
            f"Below are {k} versions of the function, each one better than the one before.")
    return HEADER.format(k=k, shown_line=line) + "\n\n```python\nimport numpy as np\n\n" + shown + "\n```\n"


def extract(reply: str) -> str | None:
    """The last function in the reply's code, renamed ``priority``; None if absent.

    Helper functions defined before it in the same block are kept, so a child
    may split its work. Imports in the block are kept too; the sandbox screens
    them.
    """
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    text = blocks[-1] if blocks else reply
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # One unparsable reply is data: pass it on so the sandbox logs a syntax failure.
        return rename(text, "priority") if "def " in text else None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    target = next((f for f in reversed(funcs) if f.name.startswith("priority")), funcs[-1])
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n is not target and not n.name.startswith("priority"))]
    parts = [ast.get_source_segment(text, n) for n in keep]
    parts.append(rename(ast.get_source_segment(text, target), "priority"))
    return "\n\n".join(p for p in parts if p) + "\n"
