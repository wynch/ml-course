"""The two fixed instructions every optimiser is measured against.

    uv run python src/baseline.py

baseline: the one-line instruction every run starts from.
by_hand:  an instruction we wrote from TREC's class definitions, before any
          optimiser ran; a person's five minutes, for scale.
"""

from __future__ import annotations

import common
import task

BY_HAND = (
    "Decide what kind of answer the question asks for. abbreviation: the question asks what an "
    "abbreviation stands for or for the short form of something. description: the question asks "
    "for a definition, an explanation, a reason or a manner (what is X, why, how does). entity: "
    "the question asks for a thing: an animal, colour, food, product, event, term, substance, "
    "sport, etc. human: the question asks for a person or a group of people, or describes who "
    "someone is. location: the question asks for a place: city, country, mountain, state. "
    "number: the question asks for a count, date, distance, money, percentage, speed, size or "
    "other number.")


def main():
    guard = common.Guard("baseline")
    ev = task.Evaluator()
    sp = task.splits()
    out = {"task_model": f"{common.TASK_ID}@{common.TASK_REVISION}",
           "dataset": f"{task.DATASET_ID}@{task.DATASET_REVISION}",
           "sizes": {k: len(v) for k, v in sp.items()},
           "label_counts": {k: [sum(r["y"] == y for r in v) for y in range(len(task.LABELS))]
                            for k, v in sp.items()},
           "template": task.TEMPLATE}
    for name, text in (("baseline", task.BASELINE), ("by_hand", BY_HAND)):
        out[name] = {"text": text}
        for split, rows in sp.items():
            r = ev.score(text, rows)
            out[name][split] = r["acc"]
            out[name][f"{split}_right"] = r["right"]
            out[name][f"{split}_pred"] = r["pred"]
    out["guard"] = guard.report()
    common.write_results("baseline", out)


if __name__ == "__main__":
    main()
