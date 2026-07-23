"""Exercise A — write a new @tool and give it to the agent.

Add a `unit_converter` tool that converts between a few common units, wrap it
with smolagents' @tool decorator, and hand it to a CodeAgent alongside the
existing tools. Then ask the agent a question that needs it.

Run:  uv run python ../exercises/exercise_a_unit_tool.py
Solution: ../../solutions/solution_a_unit_tool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the module's src/ importable no matter where you run this from.
SRC = Path(__file__).resolve().parent.parent / "python" / "src"
sys.path.insert(0, str(SRC))

from smolagents import CodeAgent, tool  # noqa: E402
from smol_way import calculator, kb_lookup, list_module_files, build_model  # noqa: E402


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units. Supports: km<->mi, kg<->lb, c<->f.

    Args:
        value: The numeric amount to convert.
        from_unit: Source unit, one of "km","mi","kg","lb","c","f".
        to_unit: Target unit, one of "km","mi","kg","lb","c","f".
    """
    # TODO(you): implement the conversions.
    #   - km -> mi: multiply by 0.621371 ; mi -> km: divide by 0.621371
    #   - kg -> lb: multiply by 2.20462  ; lb -> kg: divide by 2.20462
    #   - c  -> f : value * 9/5 + 32     ; f  -> c : (value - 32) * 5/9
    # Return a short string like "10 km = 6.21 mi". If the pair is unsupported,
    # return an "Error: ..." string (never raise) so the agent can recover.
    raise NotImplementedError("implement unit_converter")


def main():
    agent = CodeAgent(
        tools=[calculator, kb_lookup, list_module_files, unit_converter],
        model=build_model(),
        max_steps=4,
    )
    print(agent.run("Convert 10 kilometers to miles using the unit_converter tool."))


if __name__ == "__main__":
    main()
