"""The library way: the same task, the same models, in DSPy, optimised by dspy.GEPA.

    uv run python src/library.py compile    # GEPA over the 60 training questions
    uv run python src/library.py evaluate   # DSPy's own accuracy, before and after, on held-out
    uv run python src/library.py score      # the found instruction, through our evaluator

DSPy talks to models over an OpenAI-style HTTP API, so each stage starts two
local mlx_lm.server processes: the task model (thinking off) and the
reflection model. No API key is used. DSPy answers by generating text in its
own format and parsing it; our evaluator reads label scores instead. So
"evaluate" measures DSPy's program as DSPy runs it, and "score" puts the
instruction it found into our fixed template, which compares it with our runs.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from typing import Literal

import common
import task

TASK_PORT, REFLECT_PORT = 8091, 8092
MAX_METRIC_CALLS = 600     # a quarter of our runs' 2400: the run must fit a laptop sitting
SEED = 0


def serve(model: str, port: int, extra: list[str]) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.server", "--model", model, "--port", str(port),
         "--log-level", "WARNING", *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return proc
        except OSError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"server on port {port} did not start")


def program():
    import dspy

    class Classify(dspy.Signature):
        question: str = dspy.InputField()
        label: Literal[tuple(task.LABELS)] = dspy.OutputField()

    return dspy.Predict(Classify.with_instructions(task.BASELINE))


def examples(rows):
    import dspy

    return [dspy.Example(question=r["q"], label=task.LABELS[r["y"]]).with_inputs("question")
            for r in rows]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    import dspy

    right = getattr(pred, "label", None) == gold.label
    feedback = "Right." if right else f"Wrong. The correct label is {gold.label}."
    return dspy.Prediction(score=float(right), feedback=feedback)


def lms():
    import dspy

    task_lm = dspy.LM(f"openai/{common.TASK_ID}", api_base=f"http://127.0.0.1:{TASK_PORT}/v1",
                      api_key="local", temperature=0.0, max_tokens=200, cache=False)
    reflect_lm = dspy.LM(f"openai/{common.OPT_ID}", api_base=f"http://127.0.0.1:{REFLECT_PORT}/v1",
                         api_key="local", temperature=1.0, max_tokens=2000, cache=False)
    return task_lm, reflect_lm


def compile_stage(guard):
    import dspy

    task_lm, reflect_lm = lms()
    dspy.configure(lm=task_lm)
    train = examples(task.splits()["train"])
    gepa = dspy.GEPA(metric=metric, max_metric_calls=MAX_METRIC_CALLS, reflection_lm=reflect_lm,
                     num_threads=4, seed=SEED, track_stats=True)
    best = gepa.compile(program(), trainset=train, valset=train)
    det = best.detailed_results
    out = {"dspy": dspy.__version__, "max_metric_calls": MAX_METRIC_CALLS, "seed": SEED,
           "instruction": best.signature.instructions,
           "val_scores": [float(v) for v in det.val_aggregate_scores],
           "total_metric_calls": det.total_metric_calls,
           "best_idx": det.best_idx,
           "candidate_texts": [c.predictors()[0].signature.instructions for c in det.candidates],
           "parents": det.parents,
           "discovery_calls": det.discovery_eval_counts}
    out["guard"] = guard.report()
    common.write_results("library_compile", out)


def evaluate_stage(guard):
    import dspy

    task_lm, _ = lms()
    dspy.configure(lm=task_lm)
    found = common.read_results("library_compile")["instruction"]
    rows = task.splits()["heldout"]
    out = {}
    for name, text in (("baseline", task.BASELINE), ("found", found)):
        prog = dspy.Predict(program().signature.with_instructions(text))
        ev = dspy.Evaluate(devset=examples(rows), metric=lambda g, p, t=None: getattr(p, "label", None) == g.label,
                           num_threads=4, display_progress=False)
        res = ev(prog)
        out[name] = {"heldout": float(res.score) / 100}
    out["guard"] = guard.report()
    common.write_results("library_evaluate", out)


def score_stage(guard):
    ev = task.Evaluator()
    sp = task.splits()
    found = common.read_results("library_compile")["instruction"]
    out = {"instruction": found}
    for split in ("train", "heldout", "test"):
        r = ev.score(found, sp[split])
        out[split] = r["acc"]
        out[f"{split}_right"] = r["right"]
    out["guard"] = guard.report()
    common.write_results("library_score", out)


def main(stage: str):
    guard = common.Guard(f"library {stage}")
    if stage == "score":
        score_stage(guard)
        return
    procs = [serve(common.TASK_ID, TASK_PORT, ["--chat-template-args", '{"enable_thinking":false}'])]
    if stage == "compile":
        procs.append(serve(common.OPT_ID, REFLECT_PORT, []))
    try:
        {"compile": compile_stage, "evaluate": evaluate_stage}[stage](guard)
    finally:
        for p in procs:
            p.terminate()
            p.wait()


if __name__ == "__main__":
    main(sys.argv[1])
