"""Section 2: one MLP on FashionMNIST, trained six ways, three seeds each.

    uv run python src/train.py            # one chunk of at most 4.5 minutes; run again until "remaining": 0
    uv run python src/train.py --summary  # results/train.json from results/runs/*.json
    uv run python src/train.py --figures  # both, then redraws figures/race.png

A run trains one rule from one seed for its epochs. After each epoch it saves
its whole state (weights, optimizer, random generators) to python/.runs/, so
a chunk can stop between epochs and the next one picks up where it stopped.
When a run ends, its curve goes to results/runs/<rule>_s<seed>.json. Each
chunk appends its guard to results/runs/chunks.jsonl.
"""

from __future__ import annotations

import json
import sys
import time

import common
import torch

import rules as R

BATCH = 64
EPOCHS = 20
SEEDS = (0, 1, 2)
SETTINGS = {                         # fixed after the pilots; see results/expectations.json
    "backprop": {"lr": 1e-3},
    "checkpointing": {"lr": 1e-3},
    "mezo": {"lr": 2e-4, "eps": 1e-3},
    "forward_forward": {"lr": 3e-3, "theta": 2.0},
    "mono_forward": {"lr": 1e-3},
    "predictive_coding": {"lr": 1e-3, "T": 20, "gamma": 0.1},
}
RUN_LIST = [(rule, s, EPOCHS) for rule in SETTINGS for s in SEEDS] + [("mezo", 0, 100)]
# Added after every MeZO run above had diverged (results/expectations.json, "followup"):
# the same rule at half the rate, 40 epochs, with the weights' length logged.
FOLLOWUP = {"rule": "mezo", "tag": "mezo_lr1e-4", "settings": {"lr": 1e-4, "eps": 1e-3}, "epochs": 40}
RUN_LIST += [(FOLLOWUP["tag"], s, FOLLOWUP["epochs"]) for s in SEEDS]
CHUNK_SECONDS = 270


def run_name(rule: str, seed: int, epochs: int) -> str:
    if rule == FOLLOWUP["tag"]:
        return f"{rule}_s{seed}"
    return f"{rule}_s{seed}" + ("" if epochs == EPOCHS else f"_e{epochs}")


def build(rule: str, seed: int):
    if rule == FOLLOWUP["tag"]:
        return R.RULES[FOLLOWUP["rule"]](seed, **FOLLOWUP["settings"]), FOLLOWUP["settings"]
    return R.RULES[rule](seed, **SETTINGS[rule]), SETTINGS[rule]


def weight_norm(learner) -> float:
    ps = getattr(learner, "params", None) or [t for wb in learner.layers for t in wb]
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in ps)))


def one_epoch(learner, x, y, gen) -> tuple[float, float]:
    perm = torch.randperm(len(x), generator=gen)
    t0, losses = time.perf_counter(), []
    for i in range(0, len(x) - BATCH + 1, BATCH):
        idx = perm[i:i + BATCH]
        losses.append(learner.step(x[idx], y[idx]))
    return time.perf_counter() - t0, sum(losses) / len(losses)


def chunk() -> None:
    guard = common.Guard("train chunk")
    x_tr, y_tr, x_te, y_te = common.load_fashion()
    common.RUNS.mkdir(parents=True, exist_ok=True)
    common.CACHE.mkdir(parents=True, exist_ok=True)
    epochs_done = 0
    for rule, seed, epochs in RUN_LIST:
        name = run_name(rule, seed, epochs)
        if (common.RUNS / f"{name}.json").exists():
            continue
        state_path = common.CACHE / f"{name}.pt"
        if state_path.exists():
            st = torch.load(state_path, weights_only=False)
        else:
            st = {"learner": build(rule, seed)[0],
                  "gen": torch.Generator().manual_seed(1000 + seed), "curve": []}
        while len(st["curve"]) < epochs:
            if time.perf_counter() - guard.t0 > CHUNK_SECONDS:
                break
            secs, loss = one_epoch(st["learner"], x_tr, y_tr, st["gen"])
            acc = R.accuracy(st["learner"], x_te, y_te)
            row = {"epoch": len(st["curve"]) + 1, "train_loss": None if loss != loss else round(loss, 5),   # NaN -> null
                   "test_acc": round(acc, 4), "seconds": round(secs, 2)}
            if rule == FOLLOWUP["tag"]:
                row["weight_norm"] = round(weight_norm(st["learner"]), 3)
            st["curve"].append(row)
            print(f"  {name} epoch {len(st['curve'])}: loss {loss:.4f}, test {acc:.4f}, {secs:.1f} s", flush=True)
            torch.save(st, state_path)
            epochs_done += 1
        if len(st["curve"]) == epochs:
            common.write_results(name, {"rule": rule, "seed": seed, "epochs": epochs, "batch": BATCH,
                                        "settings": build(rule, seed)[1], "curve": st["curve"]}, common.RUNS)
            state_path.unlink()
        else:
            break
    remaining = sum(not (common.RUNS / f"{run_name(*r)}.json").exists() for r in RUN_LIST)
    g = guard.report()
    g["epochs_trained"] = epochs_done
    with open(common.RUNS / "chunks.jsonl", "a") as f:
        f.write(json.dumps(g) + "\n")
    print(json.dumps({"remaining": remaining}))


def summary() -> dict:
    out = {"data": "FashionMNIST, all 60,000 training images, scored on the 10,000 test images",
           "sizes": list(R.SIZES), "batch": BATCH, "epochs": EPOCHS, "seeds": list(SEEDS),
           "settings": SETTINGS, "rules": {}}
    for rule in SETTINGS:
        runs = [common.read_results(run_name(rule, s, EPOCHS), common.RUNS) for s in SEEDS]
        finals = [r["curve"][-1]["test_acc"] for r in runs]
        best = [max(c["test_acc"] for c in r["curve"]) for r in runs]
        sec = [sum(c["seconds"] for c in r["curve"]) / EPOCHS for r in runs]
        mean_curve = [round(sum(r["curve"][e]["test_acc"] for r in runs) / len(runs), 4) for e in range(EPOCHS)]
        first85 = [next((c["epoch"] for c in r["curve"] if c["test_acc"] >= 0.85), None) for r in runs]
        out["rules"][rule] = {
            "final_test_acc": finals, "final_mean": round(sum(finals) / 3, 4),
            "final_spread": round(max(finals) - min(finals), 4),
            "best_mean": round(sum(best) / 3, 4),
            "seconds_per_epoch": [round(s, 2) for s in sec], "seconds_per_epoch_mean": round(sum(sec) / 3, 2),
            "mean_curve": mean_curve, "first_epoch_at_85": first85,
            "final_train_loss": [r["curve"][-1]["train_loss"] for r in runs],
        }
    bp = out["rules"]["backprop"]
    for rule, r in out["rules"].items():
        r["gap_to_backprop_points"] = round(100 * (r["final_mean"] - bp["final_mean"]), 2)
        r["time_vs_backprop"] = round(r["seconds_per_epoch_mean"] / bp["seconds_per_epoch_mean"], 2)
    for rule in SETTINGS:
        runs = [common.read_results(run_name(rule, s, EPOCHS), common.RUNS) for s in SEEDS]
        out["rules"][rule]["best_epoch"] = [max(r["curve"], key=lambda c: c["test_acc"])["epoch"] for r in runs]
        out["rules"][rule]["nan_from_epoch"] = [next((c["epoch"] for c in r["curve"] if c["train_loss"] is None),
                                                     None) for r in runs]
    fu = [common.read_results(run_name(FOLLOWUP["tag"], s, FOLLOWUP["epochs"]), common.RUNS) for s in SEEDS]
    out["followup"] = {"note": "added after the registered MeZO runs diverged; not part of the registered comparison",
                       "settings": FOLLOWUP["settings"], "epochs": FOLLOWUP["epochs"],
                       "curves": [[c["test_acc"] for c in r["curve"]] for r in fu],
                       "weight_norms": [[c["weight_norm"] for c in r["curve"]] for r in fu],
                       "final": [r["curve"][-1]["test_acc"] for r in fu],
                       "final_mean": round(sum(r["curve"][-1]["test_acc"] for r in fu) / len(fu), 4),
                       "best_mean": round(sum(max(c["test_acc"] for c in r["curve"]) for r in fu) / len(fu), 4),
                       "nan_from_epoch": [next((c["epoch"] for c in r["curve"] if c["train_loss"] is None), None)
                                          for r in fu]}
    long = common.read_results(run_name("mezo", 0, 100), common.RUNS)
    out["mezo_long"] = {"epochs": 100, "seed": 0, "curve": [c["test_acc"] for c in long["curve"]],
                        "final": long["curve"][-1]["test_acc"], "best": max(c["test_acc"] for c in long["curve"]),
                        "seconds": round(sum(c["seconds"] for c in long["curve"]), 1),
                        "nan_from_epoch": next((c["epoch"] for c in long["curve"] if c["train_loss"] is None), None)}
    ck = [json.loads(l) for l in (common.RUNS / "chunks.jsonl").read_text().splitlines() if l.strip()]
    out["chunks"] = {"n": len(ck), "minutes_total": round(sum(c["minutes"] for c in ck), 2),
                     "longest_minutes": max(c["minutes"] for c in ck),
                     "peak_rss_gb": max(c["peak_rss_gb"] for c in ck),
                     "peak_mps_driver_gb": max(c["peak_mps_driver_gb"] for c in ck),
                     "machine": ck[-1]["machine"], "torch": ck[-1]["torch"], "python": ck[-1]["python"],
                     "threads": ck[-1]["threads"], "mps_high_watermark_ratio": ck[-1]["mps_high_watermark_ratio"]}
    common.write_results("train", out)
    for rule, r in out["rules"].items():
        print(f"{rule:18s} {r['final_mean']:.4f} ± {r['final_spread']:.4f}  {r['seconds_per_epoch_mean']:6.2f} s/epoch")
    return out


if __name__ == "__main__":
    if "--summary" in sys.argv or "--figures" in sys.argv:
        summary()
        if "--figures" in sys.argv:
            import figures
            figures.race()
    else:
        chunk()
