"""Shared tools for module 10.

Three small, *local*, offline-safe tools that both the hand-rolled ReAct loop
and the smolagents CodeAgent use:

  - ``calculator``       : evaluate a Python arithmetic expression, safely
  - ``kb_lookup``        : look a fact up in the module's shipped knowledge base
  - ``list_module_files``: list files inside the module's own directory

Each tool is a plain Python function with a clear docstring and string I/O so
the hand-rolled loop can call it directly. ``smol_way.py`` wraps the same three
functions with smolagents' ``@tool`` decorator — one source of truth, two
agent frameworks.

The calculator is deliberately *not* ``eval``. It walks the Python AST and only
permits a whitelist of arithmetic nodes and a few math functions, so a malicious
or hallucinated expression like ``__import__('os').system('rm -rf /')`` raises
instead of running.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path

# --- module paths ---------------------------------------------------------
# tools.py lives at .../modules/10-agents/python/src/tools.py
_SRC_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SRC_DIR.parent.parent  # .../modules/10-agents
_KB_PATH = _MODULE_DIR / "knowledge_base.json"


# --- calculator -----------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unsupported function: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword args are not supported")
        return fn(*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Supports + - * / // % ** , parentheses, the constants pi/e/tau, and the
    functions sqrt log log2 log10 exp sin cos tan floor ceil abs round min max.
    Example: calculator("2 ** 10 + sqrt(144)") -> "1036.0".

    Args:
        expression: A Python arithmetic expression as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # noqa: BLE001 - report any parse/eval error to the agent
        return f"Error: could not evaluate {expression!r} ({exc})"
    # Present integers without a trailing .0 so the model can copy them cleanly.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- knowledge base -------------------------------------------------------
def _load_kb() -> dict[str, str]:
    with open(_KB_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def kb_lookup(key: str) -> str:
    """Look up a fact in the local knowledge base by key.

    Keys are short phrases like "speed of light", "seconds in a day",
    "capital of japan". Matching is case-insensitive. If the exact key is
    missing, close keys are suggested so the agent can retry.

    Args:
        key: The fact to look up, e.g. "speed of light".
    """
    kb = _load_kb()
    # Be forgiving about how a small model phrases the key: lowercase, turn
    # hyphens/underscores into spaces, drop apostrophes, and collapse runs of
    # whitespace. "Seconds-in-a-day" and "seconds in a day" then match.
    def _norm(s: str) -> str:
        s = s.strip().lower().replace("-", " ").replace("_", " ").replace("'", "")
        return " ".join(s.split())
    norm = _norm(key)
    kb = {_norm(k): v for k, v in kb.items()}
    if norm in kb:
        return kb[norm]
    # Offer a soft fallback: any key that shares a word with the query.
    query_words = set(norm.split())
    near = [k for k in kb if query_words & set(k.split())]
    if near:
        return f"No exact entry for {key!r}. Did you mean: {', '.join(sorted(near))}?"
    return (
        f"No entry for {key!r}. Known keys include: "
        f"{', '.join(sorted(kb)[:6])}, ..."
    )


# --- file listing ---------------------------------------------------------
def list_module_files(subdir: str = "") -> str:
    """List files inside the module 10 directory (this course module).

    Use this to discover what materials ship with the module. Pass an empty
    string for the top level, or a subdirectory name like "figures",
    "exercises", "solutions", or "python".

    Args:
        subdir: Subdirectory under the module root, or "" for the root.
    """
    base = (_MODULE_DIR / subdir).resolve()
    # Guard against path traversal: stay inside the module directory.
    if _MODULE_DIR not in base.parents and base != _MODULE_DIR:
        return f"Error: {subdir!r} is outside the module directory"
    if not base.exists():
        return f"Error: no such subdirectory {subdir!r}"
    entries = []
    for p in sorted(base.iterdir()):
        if p.name.startswith(".") or p.name == "__pycache__":
            continue
        entries.append(p.name + ("/" if p.is_dir() else ""))
    listing = ", ".join(entries) if entries else "(empty)"
    return f"{subdir or '.'}: {listing}"


# --- registry for the hand-rolled loop ------------------------------------
# name -> (callable, one-line signature+description for the prompt)
TOOL_REGISTRY = {
    "calculator": (
        calculator,
        'calculator(expression: str) -> evaluate arithmetic, e.g. calculator("3 * (4 + 5)")',
    ),
    "kb_lookup": (
        kb_lookup,
        'kb_lookup(key: str) -> look a fact up in the local knowledge base, e.g. kb_lookup("speed of light")',
    ),
    "list_module_files": (
        list_module_files,
        'list_module_files(subdir: str) -> list files in the module dir, e.g. list_module_files("figures")',
    ),
}
