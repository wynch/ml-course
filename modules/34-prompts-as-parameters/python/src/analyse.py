"""Every number in the README, from the committed results; no model is loaded.

    uv run python src/analyse.py            # writes results/analysis.json
    uv run python src/analyse.py --figures  # and redraws the figures
"""

from __future__ import annotations

import math
import sys

import numpy as np

import common
import optimise
import task


def sign_test(a: list[int], b: list[int]) -> dict:
    """Exact two-sided sign test on paired right/wrong vectors: does b beat a?"""
    a, b = np.array(a), np.array(b)
    wins = int(((b == 1) & (a == 0)).sum())
    losses = int(((b == 0) & (a == 1)).sum())
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"wins": wins, "losses": losses, "p": p}


def quoted(text: str, rows: list[dict], n: int = 5) -> int:
    """How many questions have a run of n words, case and punctuation aside,
    copied into the instruction: a sign it learned examples, not a rule."""
    import re

    def words(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()

    t = " ".join(words(text))
    count = 0
    for r in rows:
        w = words(r["q"])
        if len(w) >= n and any(" ".join(w[i:i + n]) in t for i in range(len(w) - n + 1)):
            count += 1
    return count


def per_label(pred: list[int], rows: list[dict]) -> list[float]:
    """Share right on each label's questions, in task.LABELS order."""
    out = []
    for y in range(len(task.LABELS)):
        idx = [i for i, r in enumerate(rows) if r["y"] == y]
        out.append(sum(pred[i] == y for i in idx) / len(idx))
    return out


def mean_sd(xs) -> dict:
    xs = np.array(xs, dtype=float)
    return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
            "values": [float(x) for x in xs]}


def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for m in optimise.METHODS:
        runs[m] = [common.read_results(f"runs/{m}_s{s}") for s in optimise.SEEDS
                   if (common.RUNS / f"{m}_s{s}.json").exists()]
    return runs


def main():
    base = common.read_results("baseline")
    b = base["baseline"]
    runs = load_runs()
    base_gap = b["train"] - b["heldout"]
    out = {"baseline": {k: b[k] for k in ("train", "heldout", "test")},
           "by_hand": {k: base["by_hand"][k] for k in ("train", "heldout", "test")},
           "baseline_gap": base_gap, "budget": optimise.BUDGET, "methods": {}, "runs": []}
    out["by_hand"]["test_vs_baseline"] = sign_test(b["test_right"], base["by_hand"]["test_right"])
    sp = task.splits()        # TREC from the Hugging Face cache: quotes and labels
    out["baseline"]["test_per_label"] = per_label(b["test_pred"], sp["test"])
    out["by_hand"]["test_per_label"] = per_label(base["by_hand"]["test_pred"], sp["test"])
    for m, rs in runs.items():
        if not rs:
            continue
        row = {}
        for split in ("train", "heldout", "test"):
            row[split] = mean_sd([r["best"][split] for r in rs])
        row["test_gain"] = mean_sd([r["best"]["test"] - b["test"] for r in rs])
        row["heldout_gain"] = mean_sd([r["best"]["heldout"] - b["heldout"] for r in rs])
        row["gap"] = mean_sd([r["best"]["train"] - r["best"]["heldout"] for r in rs])
        row["gap_over_baseline"] = mean_sd([r["best"]["train"] - r["best"]["heldout"] - base_gap
                                            for r in rs])
        row["candidates"] = mean_sd([r["candidates"] for r in rs])
        row["words"] = mean_sd([len(r["best"]["text"].split()) for r in rs])
        if m == "textgrad":
            # Each edit is judged against the instruction it was made from
            # (from_acc in the log), not against the previous trace entry,
            # which may be an edit that was undone.
            edits = [e for r in rs for e in common.read_jsonl(common.RUNS / f"textgrad_s{r['seed']}.jsonl")
                     if e["event"] == "step"]
            row["steps"] = len(edits)
            row["accepted"] = sum(e["accepted"] for e in edits)
            row["raised"] = sum(e["acc"] > e["from_acc"] for e in edits)
            row["tied"] = sum(e["acc"] == e["from_acc"] for e in edits)
            row["worse"] = sum(e["acc"] < e["from_acc"] for e in edits)
        if m == "gepa":
            tr = [t for r in rs for t in r["trace"]]
            row["children"] = len(tr)
            row["kept"] = sum(t["kept"] for t in tr)
            row["front_mean"] = float(np.mean([t["front"] for t in tr]))
        out["methods"][m] = row
        for r in rs:
            out["runs"].append({
                "method": m, "seed": r["seed"], "train": r["best"]["train"],
                "heldout": r["best"]["heldout"], "test": r["best"]["test"],
                "words": len(r["best"]["text"].split()), "found_at_calls": next(
                    (c[0] for c in r["curve"] if c[1] >= r["best"]["train"]), None),
                "test_vs_baseline": sign_test(b["test_right"], r["best"]["test_right"]),
                "test_per_label": per_label(r["best"]["test_pred"], sp["test"]),
                "quotes": {k: quoted(r["best"]["text"], sp[k]) for k in ("train", "heldout", "test")},
                "text": r["best"]["text"]})
    lib = {}
    for name in ("library_compile", "library_evaluate", "library_score"):
        if (common.RESULTS / f"{name}.json").exists():
            lib[name] = common.read_results(name)
    if "library_score" in lib:
        s = lib["library_score"]
        out["dspy"] = {k: s[k] for k in ("train", "heldout", "test")}
        out["dspy"]["test_vs_baseline"] = sign_test(b["test_right"], s["test_right"])
        out["dspy"]["words"] = len(s["instruction"].split())
    if "library_compile" in lib:
        c = lib["library_compile"]
        out.setdefault("dspy", {}).update({"candidates": len(c["candidate_texts"]),
                                           "metric_calls": c["total_metric_calls"],
                                           "best_val": c["val_scores"][c["best_idx"]]})
    if "library_evaluate" in lib:
        out.setdefault("dspy", {})["own_format"] = {k: v["heldout"] for k, v in
                                                    lib["library_evaluate"].items() if k != "guard"}
    out["expectations"] = score_expectations(out)
    common.write_results("analysis", out)
    if "--figures" in sys.argv:
        import figures
        figures.main()


def score_expectations(a: dict) -> dict:
    ms = a["methods"]
    if len(ms) < 4 or any(len(v["test"]["values"]) < 3 for v in ms.values()):
        return {"note": "not all 12 runs are in"}
    bt = a["baseline"]["test"]
    best_gain = max(v["test_gain"]["mean"] for v in ms.values())
    gaps = [r["train"] - r["heldout"] - a["baseline_gap"] for r in a["runs"]]
    e = {
        "E1": all(x > bt for v in ms.values() for x in v["test"]["values"]),
        "E2": best_gain >= 0.05,
        "E3": 0.03 <= float(np.mean(gaps)) <= 0.10,
        "E4": ms["gepa"]["test"]["mean"] - ms["ape"]["test"]["mean"] >= 0.02,
        "E5": ms["textgrad"]["raised"] < ms["textgrad"]["steps"] / 2,
        "E6": sum(v["test"]["sd"] >= 0.02 for v in ms.values()) >= 2,
        "E8": all(r["test"] < 0.70 for r in a["runs"]),
    }
    if "dspy" in a and "test" in a["dspy"]:
        e["E7"] = abs(a["dspy"]["test"] - ms["gepa"]["test"]["mean"]) <= 0.05
    return {k: ("met" if v else "missed") for k, v in sorted(e.items())}


if __name__ == "__main__":
    main()
