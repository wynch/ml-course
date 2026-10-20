"""Write the explorable's data block from results/*.json.

The explorable (course/explorables/30-learning-without-a-backward-pass.html)
holds one generated block between the BEGIN DATA and END DATA markers. This
script rewrites that block and nothing else, so no measured number on the
page is typed by hand.

    uv run python src/export_explorable.py           # rewrite the block
    uv run python src/export_explorable.py --check   # exit 1 if the page is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from common import MODULE, RESULTS, read_results

PAGE = MODULE.parents[1] / "explorables" / "30-learning-without-a-backward-pass.html"
BEGIN = "/* BEGIN DATA: written by modules/30-learning-without-a-backward-pass/python/src/export_explorable.py; do not edit */"
END = "/* END DATA */"


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def build() -> dict:
    T, M, C = read_results("train"), read_results("memory"), read_results("credit")
    L = read_results("library", RESULTS / "library")
    deep = str(M["depths"][-1])
    m16 = M["by_depth"][deep]["rules"]
    share = M["share_of_backprop_extra_at_16"]
    r = T["rules"]
    c8 = C["nets"]["8 hidden"]
    memory = {"depths": M["depths"], "width": M["width"], "batch": M["batch"],
              "share_of_backprop_extra_at_16": share,
              "by_depth": {d: {"params_mb": v["params_mb"], "activation_mb_one_layer": v["activation_mb_one_layer"],
                               "rules": {k: {"extra_mb": e["extra_mb"], "rss_extra_mb": e["rss_extra_mb"]}
                                         for k, e in v["rules"].items()}}
                           for d, v in M["by_depth"].items()}}
    credit = {"steps": C["steps"], "nets": {k: {"cosine": n["cosine"], "size_ratio": n["size_ratio"],
                                                "first_nonzero_step": n["first_nonzero_step"]}
                                            for k, n in C["nets"].items()}}
    train = {"rules": {k: {"mean_curve": v["mean_curve"], "final_mean": v["final_mean"], "final_spread": v["final_spread"],
                           "seconds_per_epoch_mean": v["seconds_per_epoch_mean"], "time_vs_backprop": v["time_vs_backprop"]}
                       for k, v in r.items()},
             "followup": {"curve": [round(sum(c[e] for c in T["followup"]["curves"]) / len(T["followup"]["curves"]), 4)
                                    for e in range(T["followup"]["epochs"])],
                          "final_mean": T["followup"]["final_mean"], "epochs": T["followup"]["epochs"]}}
    library = {"n_val": L["n_val"], "base_val_acc": L["base_val_acc"], "base_memory": L["base_memory"],
               **{a: {k: L[a][k] for k in ("steps", "final_val_acc", "seconds_per_step", "mps_driver_peak_gb")}
                  for a in ("lora", "mezo")}}
    memsub = (f"A wider copy of the module's MLP (hidden layers of {M['width']:,}, a batch of {M['batch']:,} random "
              f"images), three training steps, each rule in a fresh process. <code>live tensors</code> counts every "
              f"byte torch holds in tensors; <code>whole process</code> is the resident memory the operating system "
              f"saw. Both are shown above an inference-only run at the same depth.")
    credsub = (f"Two networks at their initial weights, 256 test images. After T steps of inference, the weight "
               f"update predictive coding would make is compared with backprop's gradient, layer by layer. "
               f"Each layer's error arrives one step after the layer above it, about ten times smaller.")
    fmsub = (f"FashionMNIST, all 60,000 training images, 20 epochs, batch 64, 3 seeds, on the CPU. Then MeZO against "
             f"LoRA on SmolLM2-360M-Instruct, classifying SST-2 sentences as positive or negative.")
    notice = [
        "In section 1, MeZO's bar is empty at every batch size: it holds the weights and one layer, like inference. "
        "It is also the slowest to find the spiral: one random direction per step says little about "
        "where to move 1,218 weights.",
        "Switch the optimizer to plain SGD: Adam's two numbers per weight vanish from four bars. In the module's "
        f"measurement at {deep} hidden layers, Adam's buffers were {2 * M['by_depth'][deep]['params_mb']:.0f} MiB of "
        f"backprop's {m16['backprop']['extra_mb']:.0f}.",
        f"In section 2, Mono-Forward with plain SGD stays flat as the network deepens ({m16['mono_forward_sgd']['extra_mb']:.0f} "
        f"MiB at {deep} layers); backprop with plain SGD grows by one layer of activations per layer "
        f"({m16['backprop_sgd']['extra_mb']:.0f} MiB). Predictive coding needs the most: "
        f"{m16['predictive_coding']['extra_mb']:.0f} MiB, {share['predictive_coding']:.1f} times backprop.",
        f"Checkpointing kept {int(share['checkpointing'] * 100 + 0.5)}% of backprop's extra memory, not less: its saved "
        f"activations shrank, but gradients and Adam's buffers did not.",
        f"In section 3, drag T from 0: the 8-hidden network's lowest layer gets nothing until step "
        f"{c8['first_nonzero_step'][0]}, then an update {c8['size_ratio'][0][c8['first_nonzero_step'][0]]:.0e} times "
        f"backprop's size, whose direction is float32 rounding noise (cosine {c8['cosine'][0][c8['first_nonzero_step'][0]]:.2f}).",
        f"In section 4, predictive coding ends {pct(r['predictive_coding']['final_mean'])} against backprop's "
        f"{pct(r['backprop']['final_mean'])}, at {r['predictive_coding']['time_vs_backprop']:.1f} times the time per "
        f"epoch. MeZO peaked near {pct(r['mezo']['best_mean'])} and then diverged in every seed; at half the rate "
        f"(a follow-up) it did not, and reached {pct(T['followup']['final_mean'])} after {T['followup']['epochs']} epochs.",
        f"On the language model, MeZO moved from {pct(L['base_val_acc'])} to {pct(L['mezo']['final_val_acc'])} in "
        f"{L['mezo']['steps']:,} steps; LoRA with backprop reached {pct(L['lora']['final_val_acc'])} in "
        f"{L['lora']['steps']} steps.",
    ]
    return {"memory": memory, "credit": credit, "train": train, "library": library,
            "memsub": memsub, "credsub": credsub, "fmsub": fmsub, "notice": notice}


def render() -> str:
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":"))
    return f"{BEGIN}\nconst DATA = {data};\n{END}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    page = PAGE.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(page):
        print("markers not found in", PAGE)
        return 1
    new = pat.sub(lambda _: render(), page)
    if a.check:
        if new != page:
            print("explorable data block is stale: run src/export_explorable.py")
            return 1
        print("explorable in sync")
        return 0
    PAGE.write_text(new)
    print(f"  wrote the data block of {PAGE.name} ({len(render()) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
