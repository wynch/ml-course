"""Observability / eval mini-lab: an agent is a stochastic system — measure it.

We run the hand-rolled agent N times on a small fixed task suite and record,
per task, the success rate and the number of steps each run took. Crucially we
run at **temperature > 0** here (sampling), so the same task can succeed on one
run and fail on the next — exactly the point. A single "it worked!" demo tells
you almost nothing; a success-rate bar does.

    uv run python src/eval_lab.py            # N=10 (default)
    uv run python src/eval_lab.py --n 5      # faster

Writes ../transcripts/eval_results.json, which src/figures.py turns into
eval_success.png and eval_steps.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from handrolled import run_react
import model_local

MODULE_DIR = Path(__file__).resolve().parent.parent.parent


def _contains(*needles):
    def check(answer: str) -> bool:
        a = (answer or "").lower()
        return any(n.lower() in a for n in needles)
    return check


# Five small tasks, each with a success predicate on the final answer string.
# Kept to 1–2 hops so 50 episodes finish in a reasonable time on a laptop.
SUITE = [
    {"name": "calc_power", "task": "Use the calculator to compute 2 ** 10.",
     "ok": _contains("1024")},
    {"name": "calc_expr", "task": "Use the calculator to compute (5 + 7) * 3.",
     "ok": _contains("36")},
    {"name": "kb_light", "task": "Look up the speed of light in the knowledge base.",
     "ok": _contains("299792458")},
    {"name": "kb_capital", "task": "Look up the capital of japan in the knowledge base.",
     "ok": _contains("tokyo")},
    {"name": "two_hop", "task": "Look up 'seconds in an hour' in the knowledge base, "
                                "then multiply it by 2 using the calculator.",
     "ok": _contains("7200")},
]


def run_eval(n: int = 10, temperature: float = 0.5, max_steps: int = 4) -> dict:
    model_local.load()  # warm the weights once
    results = {"n_runs": n, "temperature": temperature, "tasks": []}
    for spec in SUITE:
        successes, steps_list = 0, []
        for i in range(n):
            tr = run_react(spec["task"], max_steps=max_steps, temperature=temperature)
            ok = tr.ok and spec["ok"](tr.answer or "")
            successes += int(ok)
            steps_list.append(tr.n_model_calls)
            print(f"[{spec['name']}] run {i+1}/{n}: "
                  f"{'PASS' if ok else 'FAIL'} in {tr.n_model_calls} steps "
                  f"-> {(tr.answer or '')[:50]!r}")
        results["tasks"].append({
            "name": spec["name"],
            "task": spec["task"],
            "success_rate": successes / n,
            "steps_per_run": steps_list,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    results = run_eval(n=args.n, temperature=args.temperature)
    results["wall_seconds"] = round(time.time() - t0, 1)

    out = MODULE_DIR / "transcripts" / "eval_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for t in results["tasks"]:
        print(f"  {t['name']:12s}  {t['success_rate']*100:5.0f}%  "
              f"(steps {min(t['steps_per_run'])}-{max(t['steps_per_run'])})")
    print(f"  wall time: {results['wall_seconds']}s over {args.n} runs x {len(SUITE)} tasks")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
