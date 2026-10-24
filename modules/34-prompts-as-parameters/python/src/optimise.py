"""Four prompt optimisers, written from scratch, on one budget.

    uv run python src/optimise.py ape 0        # one chunk of APE, seed 0

Each run tunes the instruction of the frozen task model (task.Evaluator) and
may score at most BUDGET training questions in all: a full pass over the 60
training questions costs 60, a minibatch of 6 costs 6. A repeated instruction
is looked up, not scored again. When the budget is spent, the run picks the
instruction with the best training accuracy (earliest on ties), scores it on
the held-out and test sets, and writes results/runs/<method>_s<seed>.json.

A run stops after common.CHUNK_MINUTES and resumes from its state file.

    ape       Zhou et al. (2022): write instructions from demonstrations, keep the best,
              then ask for rewordings of the top few.
    opro      Yang et al. (2023): show the optimiser the scored history, ask for better ones.
    textgrad  Yuksekgonul et al. (2024), in spirit: a critique of the errors is the
              "gradient", a rewrite that follows it is the "step"; a step that lowers
              training accuracy is undone.
    gepa      Agrawal et al. (2025), in spirit: reflect on a minibatch of answers with
              feedback, propose a child, keep it if it wins the minibatch; choose
              parents from the Pareto front over training questions.
"""

from __future__ import annotations

import hashlib
import pickle
import random
import re
import sys

import numpy as np

import common
import task

BUDGET = 2400                 # training questions scored per run: 40 full passes
SEEDS = (0, 1, 2)
METHODS = ("ape", "opro", "textgrad", "gepa")
MAX_STEPS = 400               # a run whose proposals all repeat must still end
MAX_CHARS = 1200              # longer replies are cut: the instruction must fit the context
ROSTER = "abbreviation, entity, description, human, location, number"

# ---- the optimiser's prompts (our wording) ------------------------------------------------

APE_PROMPT = """Someone was given an instruction, then labelled each question below with one word. Their labels:

{demos}

The label is always one of: {roster}.
Write the instruction they were most likely given, so that a new reader would label new questions the same way. Reply with the instruction only, in at most 120 words."""

APE_REWORD = """Rewrite this instruction in other words, keeping its meaning. Reply with the new instruction only.

Instruction: {instruction}"""

OPRO_PROMPT = """You write instructions for a small language model. The model reads the instruction and one question, then answers with one word naming the type of thing the question asks for: {roster}.

Below are instructions written so far, each with its accuracy on 60 training questions, sorted from worst to best.

{history}

Some questions with their correct labels:
{demos}

Write a new instruction that differs from all of the above and that you expect to score higher than all of them. Reply with the instruction only, in at most 120 words."""

TG_CRITIQUE = """A small language model was given this instruction:

<instruction>
{instruction}
</instruction>

For each question it answered with one word naming the type of thing asked for ({roster}). It got these wrong:

{errors}

In a few short points, say what in the instruction led to these mistakes and how the instruction should change. Do not rewrite the instruction."""

TG_STEP = """Here is an instruction for a small language model, and feedback on it.

<instruction>
{instruction}
</instruction>

<feedback>
{critique}
</feedback>

Rewrite the instruction so that it acts on the feedback. Keep what works. Reply with the new instruction only, in at most 150 words."""

GEPA_REFLECT = """I gave a small language model this instruction:

<instruction>
{instruction}
</instruction>

Here are some questions, the model's one-word answer, and feedback on each:

{cases}

The possible labels are: {roster}. Read the cases, find what the instruction misses or gets wrong, and write a better instruction. Keep the rules that worked. Reply with the new instruction only, in at most 150 words."""


def clean(reply: str) -> str:
    """The instruction inside an optimiser reply: no fences, quotes or 'Instruction:' label."""
    t = reply.strip()
    t = re.sub(r"^```\w*|```$", "", t).strip()
    t = re.sub(r"^\**(new |improved |revised )?instruction\**\s*:\s*", "", t, flags=re.I).strip()
    m = re.search(r"<instruction>(.*?)</instruction>", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t[:MAX_CHARS]


def key(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()[:12]


# ---- one run ------------------------------------------------------------------------------

class Run:
    """Budget accounting, the score cache, the log, and the state file."""

    def __init__(self, method: str, seed: int):
        self.method, self.seed = method, seed
        self.name = f"{method}_s{seed}"
        self.log_path = common.RUNS / f"{self.name}.jsonl"
        self.state_path = common.RUNS / f"{self.name}.state.pkl"
        self.train = task.splits()["train"]
        if self.state_path.exists():
            self.s = pickle.loads(self.state_path.read_bytes())
        else:
            self.s = {"calls": 0, "step": 0, "pool": {}, "order": [], "rng": random.Random(seed),
                      "extra": {}, "chunks": []}
        self.ev = None
        self.opt = None

    # The models load only when a chunk has work to do.
    def models(self):
        if self.ev is None:
            self.ev = task.Evaluator()
            self.opt = common.Optimizer()
        return self.ev, self.opt

    @property
    def left(self) -> int:
        return BUDGET - self.s["calls"]

    def save(self):
        self.state_path.write_bytes(pickle.dumps(self.s))

    def log(self, row: dict):
        row = {"step": self.s["step"], "calls": self.s["calls"], **row}
        common.append_jsonl(self.log_path, row)

    def sample(self, prompt: str, n: int) -> list[str]:
        _, opt = self.models()
        self.s["draws"] = self.s.get("draws", 0) + 1
        replies = opt.sample(prompt, n, seed=self.seed * 1_000_000 + self.s["draws"])
        return [c for c in (clean(r) for r in replies) if c]

    def full(self, text: str, parent: str | None = None, note: str = "") -> dict | None:
        """Score an instruction on all 60 training questions, once; None if out of budget."""
        k = key(text)
        if k in self.s["pool"]:
            self.log({"event": "repeat", "id": k, "parent": parent, "note": note})
            return self.s["pool"][k]
        if self.left < len(self.train):
            return None
        ev, _ = self.models()
        r = ev.score(text, self.train)
        self.s["calls"] += len(self.train)
        cand = {"id": k, "text": text, "acc": r["acc"], "right": r["right"], "pred": r["pred"],
                "parent": parent,
                "step": self.s["step"], "note": note}
        self.s["pool"][k] = cand
        self.s["order"].append(k)
        self.log({"event": "full", "id": k, "parent": parent, "acc": r["acc"], "note": note,
                  "text": text})
        return cand

    def mini(self, text: str, idx: list[int]) -> list[int] | None:
        """Score an instruction on a few training questions; None if out of budget."""
        if self.left < len(idx):
            return None
        ev, _ = self.models()
        r = ev.score(text, [self.train[i] for i in idx])
        self.s["calls"] += len(idx)
        return r["right"]

    def best(self) -> dict:
        pool = [self.s["pool"][k] for k in self.s["order"]]
        return max(pool, key=lambda c: c["acc"])   # max keeps the first of equals


def demo_lines(rows: list[dict]) -> str:
    return "\n".join(f"Question: {r['q']} -> {task.LABELS[r['y']]}" for r in rows)


# ---- APE --------------------------------------------------------------------------------

APE_FIRST = 32        # instructions written from demonstrations
APE_TOP = 4           # then rewordings of the best 4, until the budget runs out


def step_ape(run: Run) -> bool:
    s, rng = run.s, run.s["rng"]
    if s["step"] == 0:
        run.full(task.BASELINE, note="baseline")
    if s["extra"].get("written", 0) < APE_FIRST:
        # One demonstration set per instruction, 6 questions each.
        n = min(8, APE_FIRST - s["extra"].get("written", 0))
        prompts = [APE_PROMPT.format(demos=demo_lines(rng.sample(run.train, 6)), roster=ROSTER)
                   for _ in range(n)]
        texts = []
        for p in prompts:
            texts += run.sample(p, 1)
        s["extra"]["written"] = s["extra"].get("written", 0) + n
        for t in texts:
            if run.full(t, note="written") is None:
                return False
        return True
    top = sorted((s["pool"][k] for k in s["order"]), key=lambda c: -c["acc"])[:APE_TOP]
    for c in top:
        for t in run.sample(APE_REWORD.format(instruction=c["text"]), 2):
            if run.full(t, parent=c["id"], note="reworded") is None:
                return False
    return True


# ---- OPRO -------------------------------------------------------------------------------

OPRO_PER_STEP = 4
OPRO_SHOWN = 20       # the best 20 scored instructions go into the prompt


def step_opro(run: Run) -> bool:
    s, rng = run.s, run.s["rng"]
    if s["step"] == 0:
        run.full(task.BASELINE, note="baseline")
    pool = sorted((s["pool"][k] for k in s["order"]), key=lambda c: c["acc"])[-OPRO_SHOWN:]
    history = "\n\n".join(f"Instruction: {c['text']}\nAccuracy: {round(100 * c['acc'])}%"
                          for c in pool)
    prompt = OPRO_PROMPT.format(history=history, demos=demo_lines(rng.sample(run.train, 3)),
                                roster=ROSTER)
    for t in run.sample(prompt, OPRO_PER_STEP):
        if run.full(t, note="proposed") is None:
            return False
    return True


# ---- TextGrad-style ---------------------------------------------------------------------

TG_BATCH = 12         # questions drawn per step; the critique sees the wrong ones


def step_textgrad(run: Run) -> bool:
    s, rng = run.s, run.s["rng"]
    if s["step"] == 0:
        cur = run.full(task.BASELINE, note="baseline")
        s["extra"]["current"] = cur["id"]
        s["extra"]["trace"] = [{"step": 0, "acc": cur["acc"], "accepted": True}]
    cur = s["pool"][s["extra"]["current"]]
    idx = rng.sample(range(len(run.train)), TG_BATCH)
    wrong = [i for i in idx if not cur["right"][i]]
    if not wrong:
        run.log({"event": "no_errors", "id": cur["id"]})
        return True
    # The current instruction's answers are known from its full pass: no budget spent.
    preds = [cur["pred"][i] for i in wrong]
    errors = "\n".join(f"Question: {run.train[i]['q']}\nModel said: {task.LABELS[p]}\n"
                       f"Correct: {task.LABELS[run.train[i]['y']]}" for i, p in zip(wrong, preds))
    critique = run.sample(TG_CRITIQUE.format(instruction=cur["text"], errors=errors,
                                             roster=ROSTER), 1)
    if not critique:
        return True
    new = run.sample(TG_STEP.format(instruction=cur["text"], critique=critique[0]), 1)
    if not new:
        return True
    cand = run.full(new[0], parent=cur["id"], note="step")
    if cand is None:
        return False
    accepted = cand["acc"] >= cur["acc"]
    if accepted:
        s["extra"]["current"] = cand["id"]
    s["extra"]["trace"].append({"step": s["step"], "acc": cand["acc"], "from_acc": cur["acc"],
                                "accepted": accepted,
                                "critique": critique[0][:1500]})
    run.log({"event": "step", "id": cand["id"], "accepted": accepted, "acc": cand["acc"],
             "from_acc": cur["acc"], "critique": critique[0][:1500]})
    return True


# ---- GEPA-style -------------------------------------------------------------------------

GEPA_MINI = 6


def pareto_parents(pool: list[dict]) -> dict[str, int]:
    """Candidates that are best on at least one training question, with how many.

    A candidate dominated by another (no better anywhere, worse somewhere) is
    dropped. Questions nobody gets right count for no one.
    """
    R = np.array([c["right"] for c in pool])
    best = R.max(axis=0)
    wins = {}
    for j, c in enumerate(pool):
        n = int(((R[j] == best) & (best > 0)).sum())
        if n == 0:
            continue
        dominated = any((R[k] >= R[j]).all() and (R[k] > R[j]).any() for k in range(len(pool)) if k != j)
        if not dominated:
            wins[c["id"]] = n
    if not wins:   # everyone wrong everywhere: fall back to the best average
        top = max(pool, key=lambda c: c["acc"])
        wins = {top["id"]: 1}
    return wins


def step_gepa(run: Run) -> bool:
    s, rng = run.s, run.s["rng"]
    if s["step"] == 0:
        run.full(task.BASELINE, note="baseline")
        s["extra"]["trace"] = []
    pool = [s["pool"][k] for k in s["order"]]
    wins = pareto_parents(pool)
    ids = sorted(wins)
    parent = s["pool"][rng.choices(ids, weights=[wins[i] for i in ids])[0]]
    idx = rng.sample(range(len(run.train)), GEPA_MINI)
    preds = [parent["pred"][i] for i in idx]   # known from its full pass: no budget spent
    cases = "\n\n".join(
        f"Question: {run.train[i]['q']}\nModel answer: {task.LABELS[p]}\nFeedback: "
        + ("Right." if p == run.train[i]["y"] else f"Wrong. The correct label is {task.LABELS[run.train[i]['y']]}.")
        for i, p in zip(idx, preds))
    child = run.sample(GEPA_REFLECT.format(instruction=parent["text"], cases=cases, roster=ROSTER), 1)
    if not child:
        return True
    k = key(child[0])
    if k in s["pool"]:
        run.log({"event": "repeat", "id": k, "parent": parent["id"]})
        return True
    got = run.mini(child[0], idx)
    if got is None:
        return False
    before = sum(parent["right"][i] for i in idx)
    kept = sum(got) > before
    s["extra"]["trace"].append({"step": s["step"], "parent": parent["id"], "front": len(wins),
                                "mini_before": before, "mini_after": sum(got), "kept": kept})
    run.log({"event": "mini", "parent": parent["id"], "front": len(wins), "mini_before": before,
             "mini_after": sum(got), "kept": kept, "text": child[0]})
    if kept and run.full(child[0], parent=parent["id"], note="child") is None:
        return False
    return True


STEPS = {"ape": step_ape, "opro": step_opro, "textgrad": step_textgrad, "gepa": step_gepa}


def finish(run: Run):
    ev, _ = run.models()
    sp = task.splits()
    best = run.best()
    base = run.s["pool"][key(task.BASELINE)]
    out = {"method": run.method, "seed": run.seed, "budget": BUDGET, "calls": run.s["calls"],
           "steps": run.s["step"], "candidates": len(run.s["order"]),
           "optimiser": {"model": f"{common.OPT_ID}@{common.OPT_REVISION}", "temp": 1.0,
                         "top_p": 0.95},
           "task_model": f"{common.TASK_ID}@{common.TASK_REVISION}",
           "best": {"id": best["id"], "text": best["text"], "train": best["acc"], "step": best["step"]},
           "baseline_train": base["acc"],
           "curve": [], "chunks": run.s["chunks"]}
    for split in ("heldout", "test"):
        r = ev.score(best["text"], sp[split])
        out["best"][split] = r["acc"]
        out["best"][f"{split}_right"] = r["right"]
        out["best"][f"{split}_pred"] = r["pred"]
    # Best-so-far training accuracy against budget spent, from the log.
    top = 0.0
    for row in common.read_jsonl(run.log_path):
        if row["event"] == "full":
            top = max(top, row["acc"])
            out["curve"].append([row["calls"], top, row["acc"]])   # calls: spent after this pass
    if run.method in ("textgrad", "gepa"):
        out["trace"] = run.s["extra"]["trace"]
    common.write_results(f"runs/{run.name}", out)


def main(method: str, seed: int):
    guard = common.Guard(f"optimise {method} {seed}")
    run = Run(method, seed)
    done_path = common.RUNS / f"{run.name}.json"
    if done_path.exists():
        print(f"{run.name}: done")
        return
    alive = True
    while alive and guard.minutes() < common.CHUNK_MINUTES:
        alive = STEPS[method](run)
        run.s["step"] += 1
        run.save()
        print(f"{run.name} step {run.s['step']}: {run.s['calls']}/{BUDGET} calls, "
              f"best train {run.best()['acc']:.3f}", flush=True)
        if run.left < GEPA_MINI or run.s["step"] >= MAX_STEPS:
            alive = False
    rep = guard.report()
    run.s["chunks"].append(rep)
    run.save()
    if not alive:
        finish(run)
    print(f'{{"run": "{run.name}", "remaining": {max(run.left, 0) if alive else 0}}}')


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
