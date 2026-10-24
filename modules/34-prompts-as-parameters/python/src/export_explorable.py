"""Write the explorable's data block from the committed results; no model is loaded.

    uv run python src/export_explorable.py

Replaces the JSON inside <script type="application/json" id="data"> in
explorables/34-prompts-as-parameters.html.
"""

from __future__ import annotations

import json
import re

import common
import optimise
import task

PAGE = common.MODULE.parents[1] / "explorables" / "34-prompts-as-parameters.html"
SHORT = 320      # characters of each logged instruction kept for the stepper


def events(name: str) -> list[dict]:
    """The run's log, cut down to what the stepper draws."""
    out = []
    for r in common.read_jsonl(common.RUNS / f"{name}.jsonl"):
        e = {"k": r["event"], "c": r["calls"]}
        if r["event"] == "full":
            e.update(acc=round(r["acc"], 4), t=r["text"][:SHORT] + ("…" if len(r["text"]) > SHORT else ""),
                     n=r.get("note", ""))
        elif r["event"] == "mini":
            e.update(b=r["mini_before"], a=r["mini_after"], kept=r["kept"], f=r["front"],
                     t=r["text"][:SHORT] + ("…" if len(r["text"]) > SHORT else ""))
        elif r["event"] == "step":
            e.update(ok=r["accepted"], acc=round(r["acc"], 4), fr=round(r["from_acc"], 4),
                     cr=r["critique"][:SHORT] + ("…" if len(r["critique"]) > SHORT else ""))
        else:
            continue
        out.append(e)
    return out


def main():
    a = common.read_results("analysis")
    base = common.read_results("baseline")
    data = {"labels": task.LABELS, "budget": optimise.BUDGET,
            "baseline": dict(a["baseline"]), "by_hand": dict(a["by_hand"]),
            "baseline_text": base["baseline"]["text"], "by_hand_text": base["by_hand"]["text"],
            "methods": {m: {k: v[k]["mean"] if isinstance(v.get(k), dict) else v.get(k)
                            for k in ("train", "heldout", "test", "test_gain", "gap")}
                        | {"test_sd": v["test"]["sd"]}
                        for m, v in a["methods"].items()},
            "dspy": a.get("dspy"), "runs": [], "events": {}}
    data["baseline"]["per_label"] = a["baseline"]["test_per_label"]
    data["by_hand"]["per_label"] = a["by_hand"]["test_per_label"]
    for r in a["runs"]:
        name = f"{r['method']}_s{r['seed']}"
        row = {k: r[k] for k in ("method", "seed", "train", "heldout", "test", "text", "words",
                                 "found_at_calls")}
        row["p"] = r["test_vs_baseline"]["p"]
        row["per_label"] = r["test_per_label"]
        data["runs"].append(row)
        data["events"][name] = events(name)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.read_text()
    html, n = re.subn(r'(<script type="application/json" id="data">)(.*?)(</script>)',
                      lambda m: m.group(1) + blob + m.group(3), html, flags=re.S)
    assert n == 1
    PAGE.write_text(html)
    print(f"wrote {PAGE.name}: {len(blob) / 1e3:.0f} KB of data")


if __name__ == "__main__":
    main()
