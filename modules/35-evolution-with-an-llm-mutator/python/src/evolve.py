"""The evolutionary loop, with an LLM as the mutator, and its two ablations.

Arms, all with the same budget of BUDGET sampled candidates (one evaluator
call each, except exact repeats, which are skipped):

  evolve      islands, parents drawn by score, every child scored and kept
              if valid. FunSearch's loop at toy scale.
  noselect    the LLM always sees the two seed programs (first fit, best
              fit); every child is scored; the best is kept. Sampling with a
              verifier, but no evolution: no child ever becomes a parent.
  noverifier  no evaluator inside the loop: each child is accepted as it is
              and becomes the next parent. Children are scored afterwards,
              for this README only; the loop never sees those scores.

Each run appends one row per candidate to results/runs/<arm>_s<seed>.jsonl
and stops after common.CHUNK_MINUTES; run it again to continue. Every random
choice is seeded by (seed, step), so a resumed run rebuilds the same state.

    uv run python src/evolve.py evolve --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import common
import database
import prompt
import sandbox
import task

BUDGET = 400               # sampled candidates per run
SAMPLES_PER_PROMPT = 4     # one prompt, four replies in one batch
N_ISLANDS = 4
TEMPERATURE = 0.25         # percentage points of excess
RESET_EVERY = 25           # steps (100 candidates)
ARMS = ("evolve", "noselect", "noverifier", "evolve_flat")


def run_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def seed_programs() -> list[database.Program]:
    out = []
    for i, code in enumerate([task.FIRST_FIT, task.BEST_FIT]):
        bins = task.bins_used(task.first_fit if i == 0 else task.best_fit, "evolve")
        out.append(database.Program(id=-(i + 1), code=code, excess=task.excess(bins, "evolve"),
                                    signature=tuple(bins), length=len(sandbox.normalize(code))))
    return out


def as_program(row: dict) -> database.Program:
    return database.Program(id=row["idx"], code=row["code"], excess=row["excess"],
                            signature=tuple(row["bins"]), length=row["length"])


class State:
    """What the loop knows, rebuilt from the log on every start."""

    def __init__(self, arm: str, seed: int):
        self.arm, self.seed = arm, seed
        temp = math.inf if arm == "evolve_flat" else TEMPERATURE
        self.db = database.Database(N_ISLANDS, temp, RESET_EVERY)
        self.seeds = seed_programs()
        self.db.seed(self.seeds)
        self.hashes = {sandbox.code_hash(p.code): p.id for p in self.seeds}
        self.chain = [p.code for p in self.seeds]        # noverifier: last two accepted

    def apply(self, row: dict) -> None:
        if row["status"] == "valid" and self.arm.startswith("evolve"):
            self.db.add(as_program(row), row["island"])
        if row.get("hash") and row["hash"] not in self.hashes:
            self.hashes[row["hash"]] = row["idx"]
        if self.arm == "noverifier" and row["code"]:
            self.chain = [self.chain[-1], row["code"]]

    def end_step(self, step: int) -> dict | None:
        if self.arm.startswith("evolve"):
            return self.db.maybe_reset(step, np.random.default_rng([self.seed, step, 7]))
        return None

    def parents(self, step: int) -> tuple[int | None, list]:
        """(island, parents worst first) for this step."""
        if self.arm == "noselect":
            return None, [(p.id, p.code) for p in self.seeds]
        if self.arm == "noverifier":
            return None, [(None, c) for c in self.chain]
        rng = np.random.default_rng([self.seed, step])
        island = int(rng.integers(N_ISLANDS))
        return island, [(p.id, p.code) for p in self.db.sample_parents(island, rng)]


def score_row(code: str | None, sets=("evolve",)) -> dict:
    if code is None:
        return {"status": "noparse", "error": "no function in the reply", "bins": {}, "seconds": 0.0}
    return sandbox.evaluate(code, sets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()

    name = run_name(args.arm, args.seed)
    log = common.RUNS / f"{name}.jsonl"
    guard = common.Guard(f"evolve.py {name}")
    rows = common.read_jsonl(log)
    state = State(args.arm, args.seed)
    for step in sorted({r["step"] for r in rows}):
        for r in (r for r in rows if r["step"] == step):
            state.apply(r)
        if sum(1 for r in rows if r["step"] == step) == SAMPLES_PER_PROMPT:
            state.end_step(step)
    done = len(rows)
    if done >= args.budget:
        print(f"{name}: done ({done} candidates)")
        return

    mutator = common.Mutator()
    step = done // SAMPLES_PER_PROMPT
    while done < args.budget and guard.minutes() < common.CHUNK_MINUTES:
        island, parents = state.parents(step)
        text = prompt.build([code for _, code in parents])
        replies = mutator.sample(text, SAMPLES_PER_PROMPT, seed=args.seed * 100_003 + step)
        for j, reply in enumerate(replies):
            code = prompt.extract(reply["text"])
            row = {"idx": done, "step": step, "arm": args.arm, "seed": args.seed,
                   "island": island, "parents": [pid for pid, _ in parents],
                   "code": code, "reply_tokens": reply["tokens"],
                   "gen_seconds": reply["gen_seconds"]}
            h = None
            if code is not None:
                try:
                    h = sandbox.code_hash(code)
                except SyntaxError:
                    h = None
            row["hash"] = h
            if args.arm != "noverifier" and h is not None and h in state.hashes:
                result = {"status": "duplicate", "error": f"same code as {state.hashes[h]}",
                          "bins": {}, "seconds": 0.0}
                row["duplicate_of"] = state.hashes[h]
            else:
                result = score_row(code)
            row.update(status=result["status"], error=result["error"],
                       eval_seconds=result["seconds"])
            bins = result["bins"].get("evolve")
            row["bins"] = bins
            row["excess"] = task.excess(bins, "evolve") if bins else None
            row["length"] = len(sandbox.normalize(code)) if h else None
            # The no-verifier loop accepts every child; its score is logged
            # for the README, never read by the loop.
            row["seen_by_loop"] = args.arm != "noverifier"
            row["wall"] = round(time.time(), 1)
            common.append_jsonl(log, row)
            state.apply(row)
            done += 1
            print(f"{name} #{done:3d} step {step:2d} island {island} {row['status']:9s} "
                  f"{row['excess'] if row['excess'] is not None else '':}")
        reset = state.end_step(step)
        if reset:
            common.append_jsonl(common.RUNS / f"{name}.resets.jsonl", reset)
        step += 1

    g = guard.report()
    g.update(candidates_after=done, mutator=mutator.settings)
    common.append_jsonl(common.RUNS / f"{name}.guards.jsonl", g)
    print(f"{name}: {done}/{args.budget} candidates" + ("" if done >= args.budget else ", run again"))


if __name__ == "__main__":
    main()
