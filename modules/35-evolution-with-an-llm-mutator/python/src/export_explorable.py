"""Write the explorable's data block from the logs and results/analysis.json.

    uv run python src/export_explorable.py
"""

from __future__ import annotations

import json
import re

import numpy as np

import common
import evolve
import task

PAGE = common.MODULE.parents[1] / "explorables" / "35-evolution-with-an-llm-mutator.html"
REPLAY_ITEMS = 150


def choices(code_or_fn, sizes, capacity):
    """Bin index chosen for each item (a new bin gets the next index)."""
    if callable(code_or_fn):
        fn = code_or_fn
    else:
        env = {"np": np, "numpy": np}
        exec(code_or_fn, env)
        fn = env["priority"]
    room, out = [], []
    with np.errstate(all="ignore"):
        for item in sizes:
            item = float(item)
            fits = [i for i, r in enumerate(room) if r >= item]
            if not fits:
                room.append(capacity - item)
                out.append(len(room) - 1)
                continue
            s = np.asarray(fn(item, np.array([room[i] for i in fits])), dtype=float)
            b = fits[int(np.argmax(s))]
            room[b] -= item
            out.append(b)
    return out, len(room)


def stepper(name: str) -> list[dict]:
    """One frame per prompt of a run: island, parents, children, bests."""
    arm, seed = name.rsplit("_s", 1)
    rows = common.read_jsonl(common.RUNS / f"{name}.jsonl")[: evolve.BUDGET]
    state = evolve.State(arm, int(seed))
    by_id = {p.id: p.excess for p in state.seeds}
    best = state.db.best().excess
    frames = []
    for step in sorted({r["step"] for r in rows}):
        kids = [r for r in rows if r["step"] == step]
        parents = [{"id": pid, "x": round(by_id[pid], 3)} for pid in kids[0]["parents"]]
        for r in kids:
            state.apply(r)
            if r["status"] == "valid":
                by_id[r["idx"]] = r["excess"]
                best = min(best, r["excess"])
        reset = state.end_step(step)
        frames.append({
            "step": step, "island": kids[0]["island"], "parents": parents,
            "kids": [{"id": r["idx"], "s": r["status"],
                      "x": round(r["excess"], 3) if r["excess"] is not None else None} for r in kids],
            "best": round(best, 3),
            "islands": [round(i.best().excess, 3) for i in state.db.islands],
            "reset": reset["wiped"] if reset else None,
        })
    return frames


def main() -> None:
    a = common.read_results("analysis")
    runs = a["runs"]
    best_evolve = min((r for r in runs.values() if r["arm"] == "evolve"),
                      key=lambda r: r["kept"]["excess"]["evolve"])
    best_random = min((r for r in runs.values() if r["arm"] == "random"),
                      key=lambda r: r["kept"]["excess"]["evolve"])
    sizes, cap = task.instances("heldout")[0]
    replay = {"sizes": [int(s) for s in sizes[:REPLAY_ITEMS]], "capacity": cap,
              "l1_full": task.l1_bound(sizes, cap), "heuristics": []}
    for label, src in (("first fit", task.first_fit), ("best fit", task.best_fit),
                       (f"evolved (seed {best_evolve['seed']})", best_evolve["kept"]["code"]),
                       (f"random search (seed {best_random['seed']})", best_random["kept"]["code"])):
        picks, _ = choices(src, sizes[:REPLAY_ITEMS], cap)
        _, full = choices(src, sizes, cap)
        replay["heuristics"].append({"label": label, "picks": picks, "bins_full": full})

    env = {"np": np, "numpy": np}
    exec(best_evolve["kept"]["code"], env)
    with np.errstate(all="ignore"):
        ev_total = sum(task.bins_used(env["priority"], "heldout"))
    totals = {"l1": sum(task.l1_bound(s, c) for s, c in task.instances("heldout")),
              "best_fit": sum(task.bins_used(task.best_fit, "heldout")), "evolved": ev_total,
              "seed": best_evolve["seed"]}
    nov = runs.get("noverifier_s0")
    data = {
        "budget": a["budget"], "per_prompt": a["samples_per_prompt"], "islands": a["islands"],
        "temperature": a["temperature"], "baselines": a["baselines"],
        "summary": {arm: {k: v for k, v in s.items() if k in
                          ("evolve", "heldout", "shift", "gain_evolve", "gain_heldout", "gain_shift",
                           "valid_share", "fail_share", "duplicate_share", "kept_runs")}
                    for arm, s in a["summary"].items()},
        "curves": {n: r["best_so_far"] for n, r in runs.items() if r["arm"] != "noverifier"},
        "stepper": stepper("evolve_s0"),
        "replay": replay,
        "predict_truth": ev_total,
        "totals": totals,
        "code": {"evolve": best_evolve["kept"]["code"], "random": best_random["kept"]["code"],
                 "noverifier": nov["kept"]["code"] if nov else None},
        "noverifier": {"status": nov["status_trace"], "x": nov["excess_trace"]} if nov else None,
    }
    # "</" would end the <script> element early if a program's code held it.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.read_text()
    new, n = re.subn(r'(<script type="application/json" id="data">).*?(</script>)',
                     lambda m: m.group(1) + blob + m.group(2), html, flags=re.S)
    if n != 1:
        raise SystemExit("data block not found")
    new, n = re.subn(r'data-value="[^"]*" id="predict-answer"',
                     f'data-value="{data["predict_truth"]}" id="predict-answer"', new)
    if n != 1:
        raise SystemExit("predict answer not found")
    PAGE.write_text(new)
    print(f"wrote {len(blob) / 1024:.0f} KB of data into {PAGE.name}; predict truth {data['predict_truth']}")


if __name__ == "__main__":
    main()
