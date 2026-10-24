"""Read every run log, score the kept programs on held-out sets, write results/analysis.json.

No model runs here. The only evaluator calls are the held-out and shift
scores of each run's kept program, through the same sandbox.

    uv run python src/analyse.py
"""

from __future__ import annotations

import statistics
from collections import Counter

import common
import evolve
import sandbox
import task

ARMS = {"evolve": [0, 1, 2], "noselect": [0, 1, 2], "noverifier": [0, 1, 2],
        "random": [0, 1, 2], "evolve_flat": [0]}
FAIL = ("screen", "crash", "timeout", "invalid", "noparse")


def clusters_at_prompts(arm: str, seed: int, rows: list[dict]) -> list[int]:
    """Clusters on the sampled island at each prompt, by replaying the log.

    Parents are two draws without replacement from an island's clusters, so
    with only two clusters both are always drawn and the temperature has no
    effect; it acts only from three clusters on.
    """
    import numpy as np

    state = evolve.State(arm, seed)
    out = []
    for step in sorted({r["step"] for r in rows}):
        island = int(np.random.default_rng([seed, step]).integers(evolve.N_ISLANDS))
        out.append(len(state.db.islands[island].clusters))
        logged = [r for r in rows if r["step"] == step]
        for r in logged:
            state.apply(r)
        if len(logged) == evolve.SAMPLES_PER_PROMPT:
            state.end_step(step)
    return out


def baselines() -> dict:
    out = {}
    for name, fn in (("first_fit", task.first_fit), ("best_fit", task.best_fit)):
        out[name] = {s: round(task.excess(task.bins_used(fn, s), s), 4) for s in task.SETS}
    return out


def kept(rows: list[dict], arm: str) -> dict:
    """The program the arm would hand over at the end."""
    if arm == "noverifier":
        # No evaluator: the loop ends holding its last child, whatever it is.
        last = next(r for r in reversed(rows) if r["code"])
        return {"idx": last["idx"], "code": last["code"], "how": "last child"}
    valid = [r for r in rows if r["status"] == "valid"]
    best = min(valid, key=lambda r: (r["excess"], r["length"], r["idx"]))
    return {"idx": best["idx"], "code": best["code"], "how": "best valid on evolve"}


def best_so_far(rows: list[dict], start: float) -> list[float]:
    cur, out = start, []
    for r in rows:
        if r["status"] == "valid" and r["excess"] < cur:
            cur = r["excess"]
        out.append(round(cur, 4))
    return out


def mean_sd(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 4),
            "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else None,
            "values": [round(x, 4) for x in xs]}


def main() -> None:
    guard = common.Guard("analyse.py")
    base = baselines()
    bf = base["best_fit"]
    runs = {}
    for arm, seeds in ARMS.items():
        for seed in seeds:
            name = f"{arm}_s{seed}"
            rows = common.read_jsonl(common.RUNS / f"{name}.jsonl")
            # The no-verifier run was stopped early (README, section 4); the rest need the full budget.
            if len(rows) < (evolve.BUDGET if arm != "noverifier" else 200):
                print(f"skip {name}: {len(rows)} rows")
                continue
            rows = rows[: evolve.BUDGET]
            status = Counter(r["status"] for r in rows)
            k = kept(rows, arm)
            scored = sandbox.evaluate(k["code"], sets=("evolve", "heldout", "shift"))
            k["status"] = scored["status"]
            k["excess"] = ({s: round(task.excess(b, s), 4) for s, b in scored["bins"].items()}
                           if scored["status"] == "valid" else None)
            valid = [r for r in rows if r["status"] == "valid"]
            runs[name] = {
                "arm": arm, "seed": seed, "candidates": len(rows),
                "status": dict(status),
                "valid_share": round(status["valid"] / len(rows), 4),
                "fail_share": round(sum(status[s] for s in FAIL) / len(rows), 4),
                "duplicate_share": round(status["duplicate"] / len(rows), 4),
                "evaluator_calls": len(rows) - status["duplicate"],
                "best_so_far": best_so_far(rows, bf["evolve"]),
                "posthoc_best_evolve": round(min(r["excess"] for r in valid), 4) if valid else None,
                "kept": k,
                "distinct_behaviours": len({tuple(r["bins"]) for r in valid}),
            }
            if arm == "noverifier":
                runs[name]["excess_trace"] = [r["excess"] if r["status"] == "valid" else None
                                              for r in rows]
                runs[name]["status_trace"] = [r["status"] for r in rows]
            if arm.startswith("evolve"):
                counts = clusters_at_prompts(arm, seed, rows)
                runs[name]["clusters_at_prompt"] = {
                    "median": statistics.median(counts), "max": max(counts),
                    "share_3_or_more": round(sum(c >= 3 for c in counts) / len(counts), 4)}
                minutes = [g["minutes"] for g in common.read_jsonl(common.RUNS / f"{name}.guards.jsonl")]
                runs[name]["minutes"] = round(sum(minutes), 2)
            print(f"{name}: kept {k['excess']}  status {dict(status)}")

    summary = {}
    for arm in ARMS:
        names = [n for n in runs if runs[n]["arm"] == arm]
        if not names:
            continue
        ok = [runs[n]["kept"]["excess"] for n in names if runs[n]["kept"]["excess"]]
        entry = {"runs": names,
                 "valid_share": mean_sd([runs[n]["valid_share"] for n in names]),
                 "fail_share": mean_sd([runs[n]["fail_share"] for n in names]),
                 "duplicate_share": mean_sd([runs[n]["duplicate_share"] for n in names]),
                 "kept_runs": sum(1 for n in names if runs[n]["kept"]["excess"])}
        for s in task.SETS:
            if ok:
                entry[s] = mean_sd([e[s] for e in ok])
                entry[f"gain_{s}"] = mean_sd([bf[s] - e[s] for e in ok])
        if ok:
            entry["gap_evolve_minus_heldout_gain"] = mean_sd(
                [(bf["evolve"] - e["evolve"]) - (bf["heldout"] - e["heldout"]) for e in ok])
        summary[arm] = entry

    data = {"budget": evolve.BUDGET, "samples_per_prompt": evolve.SAMPLES_PER_PROMPT,
            "islands": evolve.N_ISLANDS, "temperature": evolve.TEMPERATURE,
            "reset_every_steps": evolve.RESET_EVERY,
            "sets": {k: {kk: vv for kk, vv in v.items()} for k, v in task.SETS.items()},
            "baselines": base, "runs": runs, "summary": summary}
    data["guard"] = guard.report()
    common.write_results("analysis", data)
    common.write_results("scored", score(data))


def score(a: dict) -> dict:
    """Each line of results/expectations.json, met or missed, with the numbers."""
    bf = a["baselines"]["best_fit"]
    runs, summ = a["runs"], a["summary"]
    ev = [r["kept"]["excess"] for r in runs.values() if r["arm"] == "evolve"]
    v = {}

    def put(key, ok, numbers):
        v[key] = {"verdict": "met" if ok else "missed", "numbers": numbers}

    put("E1", all(e["evolve"] < bf["evolve"] for e in ev) and summ["evolve"]["evolve"]["mean"] <= 4.30,
        {"per_seed": [e["evolve"] for e in ev], "mean": summ["evolve"]["evolve"]["mean"]})
    d2 = summ["noselect"]["evolve"]["mean"] - summ["evolve"]["evolve"]["mean"]
    put("E2", d2 >= 0.2, {"noselect_minus_evolve": round(d2, 4)})
    d3 = summ["random"]["evolve"]["mean"] - summ["evolve"]["evolve"]["mean"]
    put("E3", d3 >= 0.2, {"random_minus_evolve": round(d3, 4)})
    keep = [(bf["heldout"] - e["heldout"]) >= 0.5 * (bf["evolve"] - e["evolve"]) for e in ev]
    put("E4", all(keep), {"kept_half": keep,
                          "gain_evolve": [round(bf["evolve"] - e["evolve"], 4) for e in ev],
                          "gain_heldout": [round(bf["heldout"] - e["heldout"], 4) for e in ev]})
    shrink = [(bf["shift"] - e["shift"]) < (bf["heldout"] - e["heldout"]) for e in ev]
    put("E5", sum(shrink) >= 2, {"shift_gain_below_heldout_gain": shrink,
                                 "gain_shift": [round(bf["shift"] - e["shift"], 4) for e in ev]})
    nov = [r for r in runs.values() if r["arm"] == "noverifier"]
    worse = [n["kept"]["excess"] is None or n["kept"]["excess"]["evolve"] > bf["evolve"] for n in nov]
    put("E6", sum(worse) >= min(2, len(nov)) and all(n["fail_share"] >= 0.2 for n in nov),
        {"end_worse_or_broken": worse, "fail_share": [n["fail_share"] for n in nov]})
    put("E7", summ["evolve"]["valid_share"]["mean"] >= 0.6, {"valid_share": summ["evolve"]["valid_share"]})
    put("E8", summ["evolve"]["duplicate_share"]["mean"] < 0.1,
        {"duplicate_share": summ["evolve"]["duplicate_share"]})
    flat = runs["evolve_flat_s0"]["kept"]["excess"]["evolve"] if "evolve_flat_s0" in runs else float("nan")
    put("E9", flat > summ["evolve"]["evolve"]["mean"], {"evolve_flat": flat,
                                                        "evolve_mean": summ["evolve"]["evolve"]["mean"]})
    met = sum(x["verdict"] == "met" for x in v.values())
    print(f"expectations: {met} met, {len(v) - met} missed")
    seeds = {arm: len(e["runs"]) for arm, e in summ.items()}
    return {"verdicts": v, "met": met, "missed": len(v) - met, "seeds_per_arm": seeds}


if __name__ == "__main__":
    main()
