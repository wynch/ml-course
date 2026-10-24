"""OpenEvolve's evaluator for module 35: our sandbox, our evolve set.

OpenEvolve maximizes combined_score, so it gets minus the excess over L1 in
percent. A program that fails gets -100, far below any valid one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sandbox  # noqa: E402
import task  # noqa: E402


def evaluate(program_path: str) -> dict:
    code = Path(program_path).read_text()
    m = re.search(r"# EVOLVE-BLOCK-START\n(.*?)# EVOLVE-BLOCK-END", code, re.S)
    body = m.group(1) if m else code
    out = sandbox.evaluate(body)
    if out["status"] != "valid":
        return {"combined_score": -100.0, "valid": 0}
    excess = task.excess(out["bins"]["evolve"], "evolve")
    return {"combined_score": -excess, "excess_percent": excess, "valid": 1}
