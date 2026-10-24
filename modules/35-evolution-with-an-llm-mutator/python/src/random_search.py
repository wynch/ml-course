"""Random search with no LLM: random expressions, same evaluator, same budget.

Each candidate is a random expression tree over the item size, the bins'
remaining room and a few constants, wrapped so it always returns one score
per bin. It goes through the same sandbox and the same dedupe as the LLM's
children, and the best valid one is kept.

    uv run python src/random_search.py --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

import common
import evolve
import sandbox
import task

TERMINALS = ["item", "bins", "(bins - item)", "0.5", "1.0", "2.0", "3.0", "10.0", "100.0"]
UNARY = ["-({})", "({})**2", "np.sqrt(np.abs({}))", "np.log1p(np.abs({}))",
         "np.exp(-({}) / 10.0)", "1.0 / (1.0 + np.abs({}))"]
BINARY = ["({} + {})", "({} - {})", "({} * {})", "({} / ({} + 1e-6))"]


def expression(rng: np.random.Generator, depth: int) -> str:
    if depth == 0 or rng.random() < 0.3:
        return TERMINALS[rng.integers(len(TERMINALS))]
    if rng.random() < 0.4:
        return UNARY[rng.integers(len(UNARY))].format(expression(rng, depth - 1))
    return BINARY[rng.integers(len(BINARY))].format(expression(rng, depth - 1),
                                                    expression(rng, depth - 1))


def program(rng: np.random.Generator, max_depth: int = 3) -> str:
    expr = expression(rng, max_depth)
    return f"def priority(item, bins):\n    return np.zeros_like(bins) + {expr}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=evolve.BUDGET)
    args = ap.parse_args()
    name = f"random_s{args.seed}"
    log = common.RUNS / f"{name}.jsonl"
    for stale in (log, common.RUNS / f"{name}.guards.jsonl"):
        stale.unlink(missing_ok=True)
    guard = common.Guard(f"random_search.py {name}")
    rng = np.random.default_rng(args.seed)
    seen: dict[str, int] = {}
    for idx in range(args.budget):
        code = program(rng)
        h = sandbox.code_hash(code)
        row = {"idx": idx, "step": idx, "arm": "random", "seed": args.seed, "island": None,
               "parents": [], "code": code, "hash": h}
        if h in seen:
            result = {"status": "duplicate", "error": f"same code as {seen[h]}", "bins": {}, "seconds": 0.0}
            row["duplicate_of"] = seen[h]
        else:
            seen[h] = idx
            result = sandbox.evaluate(code)
        bins = result["bins"].get("evolve")
        row.update(status=result["status"], error=result["error"], eval_seconds=result["seconds"],
                   bins=bins, excess=task.excess(bins, "evolve") if bins else None,
                   length=len(sandbox.normalize(code)), seen_by_loop=True)
        common.append_jsonl(log, row)
    g = guard.report()
    g.update(candidates_after=args.budget)
    common.append_jsonl(common.RUNS / f"{name}.guards.jsonl", g)
    rows = common.read_jsonl(log)
    valid = [r for r in rows if r["status"] == "valid"]
    best = min(valid, key=lambda r: r["excess"])
    print(f"{name}: {len(valid)} valid of {len(rows)}; best {best['excess']:.3f}%\n{best['code']}")


if __name__ == "__main__":
    main()
