"""Solution A — a `unit_converter` @tool added to the agent.

The tool itself is pure Python and is unit-tested offline (no model needed);
`test_solutions.py` checks the conversions. Running this file end-to-end drives
the local agent, which is slow but reproducible at temperature 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "python" / "src"
sys.path.insert(0, str(SRC))

from smolagents import CodeAgent, tool  # noqa: E402
from smol_way import calculator, kb_lookup, list_module_files, build_model  # noqa: E402

_FACTORS = {
    ("km", "mi"): lambda v: v * 0.621371,
    ("mi", "km"): lambda v: v / 0.621371,
    ("kg", "lb"): lambda v: v * 2.20462,
    ("lb", "kg"): lambda v: v / 2.20462,
    ("c", "f"): lambda v: v * 9 / 5 + 32,
    ("f", "c"): lambda v: (v - 32) * 5 / 9,
}


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units. Supports: km<->mi, kg<->lb, c<->f.

    Args:
        value: The numeric amount to convert.
        from_unit: Source unit, one of "km","mi","kg","lb","c","f".
        to_unit: Target unit, one of "km","mi","kg","lb","c","f".
    """
    a, b = from_unit.strip().lower(), to_unit.strip().lower()
    if a == b:
        return f"{value:g} {a} = {value:g} {b}"
    fn = _FACTORS.get((a, b))
    if fn is None:
        return (f"Error: unsupported conversion {from_unit}->{to_unit}. "
                f"Supported: km<->mi, kg<->lb, c<->f.")
    return f"{value:g} {a} = {fn(value):.4g} {b}"


def main():
    agent = CodeAgent(
        tools=[calculator, kb_lookup, list_module_files, unit_converter],
        model=build_model(),
        max_steps=4,
    )
    print(agent.run("Convert 10 kilometers to miles using the unit_converter tool."))


if __name__ == "__main__":
    main()
