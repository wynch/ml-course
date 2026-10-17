"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/18-beyond-attention.html) holds one
generated block between the BEGIN DATA and END DATA markers. This script
rewrites that block and nothing else, so no number on the page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, read_results

PAGE = MODULE.parents[1] / "explorables" / "18-beyond-attention.html"
BEGIN = "/* BEGIN DATA: written by modules/18-beyond-attention/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def build() -> dict:
    gru, ssm, cmp_, dec, hf = (read_results(n) for n in ("gru", "ssm", "compare", "decode", "hf_compare"))
    sel = ssm["models"]["selective"]
    measured = {dev: {k: v["at"] for k, v in kinds.items()} for dev, kinds in dec["devices"].items()}
    g2, mb = hf["models"]["gpt2"], hf["models"]["mamba"]
    g2c, mbc = g2["config"], mb["config"]
    mamba_state = mbc["num_hidden_layers"] * mbc["intermediate_size"] * (mbc["state_size"] + mbc["conv_kernel"])
    rows = [
        {"name": "GPT (attention)", "params": cmp_["models"]["gpt"]["params"], "steps_val": None,
         "budget_val": cmp_["models"]["gpt"]["run"]["val_loss"], "budget_steps": cmp_["models"]["gpt"]["run"]["steps"]},
        {"name": "GRU", "params": gru["params"], "steps_val": gru["run"]["val_loss"],
         "budget_val": cmp_["models"]["gru"]["run"]["val_loss"], "budget_steps": cmp_["models"]["gru"]["run"]["steps"]},
        {"name": "diagonal linear recurrence", "params": ssm["models"]["lti"]["params"],
         "steps_val": ssm["models"]["lti"]["run"]["val_loss"], "budget_val": None, "budget_steps": None},
        {"name": "selective SSM", "params": sel["params"], "steps_val": sel["run"]["val_loss"],
         "budget_val": cmp_["models"]["selective"]["run"]["val_loss"],
         "budget_steps": cmp_["models"]["selective"]["run"]["steps"]},
    ]
    return {
        "tiny": {
            "gpt": {"layers": 3, "d": 128, "floats_per_token": 2 * 3 * 128},
            "gru": {"layers": gru["n_layer"], "hidden": gru["hidden"], "state_floats": gru["state_floats"]},
            "selective": {"layers": sel["n_layer"], "E": sel["E"], "N": sel["N"], "state_floats": sel["state_floats"]},
            "checkpoints": [str(c) for c in dec["checkpoints"]],
            "measured": measured,
        },
        "hf": {
            "device": hf["device"],
            "contexts": list(g2["contexts"]),
            "gpt2": {"layers": g2c["n_layer"], "d": g2c["n_embd"], "floats_per_token": 2 * g2c["n_layer"] * g2c["n_embd"],
                     "cache": {t: c["cache_bytes_after_context"] for t, c in g2["contexts"].items()}},
            "mamba": {"layers": mbc["num_hidden_layers"], "E": mbc["intermediate_size"], "N": mbc["state_size"],
                      "state_floats": mamba_state,
                      "cache": {t: c["cache_bytes_after_context"] for t, c in mb["contexts"].items()}},
        },
        "loss": {"steps": gru["run"]["steps"], "budget": cmp_["budget_seconds"],
                 "committed": cmp_["committed_gpt_val_loss"], "rows": rows},
    }


def block() -> str:
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":"))
    return f"{BEGIN}\nconst DATA = {data};\n{END}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    page = PAGE.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(page):
        sys.exit(f"markers not found in {PAGE}")
    new = pattern.sub(lambda _: block(), page)
    if args.check:
        if new != page:
            sys.exit(f"{PAGE.name} is stale: run src/export_explorable.py")
        print(f"{PAGE.name} is in sync with results/")
        return
    PAGE.write_text(new)
    print(f"wrote the data block of {PAGE.name} ({len(block()) // 1024} KB)")


if __name__ == "__main__":
    main()
