"""Read every run, write the summaries the README quotes, and score each
expectation written in results/expectations.json before the runs.

    uv run python src/analyse.py            # writes results/forget.json, plasticity.json, library.json
    uv run python src/analyse.py --figures  # and redraws every figure

No model is loaded; it reads JSON only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, read_results, write_results

ARMS = ["full", "full_slow", "lora4", "lora32", "replay", "distill"]
SEEDS = (0, 1, 2)
PLAST_ARMS = ["backprop", "cbp", "sgd", "fresh"]
LIB_ARMS = ["lora8_gentle", "lora8", "lora64", "lora8_replay"]
BLOCK = 50


def runs(folder: str, name: str) -> dict:
    return json.loads((RESULTS / folder / f"{name}.json").read_text())


def r(x: float, d: int = 4) -> float:
    return round(float(x), d)


def stats(xs) -> dict:
    xs = np.asarray(xs, dtype=float)
    return {"mean": r(xs.mean()), "min": r(xs.min()), "max": r(xs.max()), "seeds": [r(x) for x in xs]}


def interp_at(xs, ys, x0: float) -> float | None:
    """y where the curve (xs, ys), walked in order, first reaches x0 by linear interpolation."""
    for i in range(1, len(xs)):
        lo, hi = sorted((xs[i - 1], xs[i]))
        if lo <= x0 <= hi and xs[i] != xs[i - 1]:
            t = (x0 - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return None


def spear(a, b) -> float:
    return r(spearmanr(a, b).statistic, 3)


# ---------------------------------------------------------------- sections 2-3

def forget() -> dict:
    exp = read_results("expectations")
    R = {a: [runs("runs", f"{a}_s{s}") for s in SEEDS] for a in ARMS}
    out = {"arms": {}, "steps": [row["step"] for row in R["full"][0]["log"]]}
    keys = ["old_loss", "new_loss", "kl_new", "kl_old"]
    for a, rs in R.items():
        logs = [rr["log"] for rr in rs]
        curve = {k: [r(np.mean([lg[i][k] for lg in logs]), 5) for i in range(len(logs[0]))] for k in keys}
        curve["forgetting"] = [r(v - curve["old_loss"][0], 5) for v in curve["old_loss"]]
        out["arms"][a] = {
            "spec": rs[0]["spec"],
            "trainable_parameters": rs[0]["trainable_parameters"],
            "final": {k: stats([rr["final"][k] for rr in rs])
                      for k in ("old_loss", "new_loss", "forgetting", "new_gain", "kl_new", "kl_old")},
            "curve_mean": curve,
            "samples_s0": rs[0]["samples"],
            "minutes": stats([rr["guard"]["minutes"] for rr in rs]),
            "peak_rss_gb": max(rr["guard"]["peak_rss_gb"] for rr in rs),
        }
    finals = [(a, rr["final"]) for a in ARMS for rr in R[a]]
    f = [x["forgetting"] for _, x in finals]
    pts = [(row["old_loss"] - rr["log"][0]["old_loss"], row["kl_new"], row["kl_old"])
           for a in ARMS for rr in R[a] for row in rr["log"][1:]]
    out["one_number"] = {
        "final_points": len(finals),
        "spearman_final_kl_new": spear(f, [x["kl_new"] for _, x in finals]),
        "spearman_final_kl_old": spear(f, [x["kl_old"] for _, x in finals]),
        "spearman_final_new_gain": spear(f, [x["new_gain"] for _, x in finals]),
        "all_points": len(pts),
        "spearman_all_kl_new": spear([p[0] for p in pts], [p[1] for p in pts]),
        "spearman_all_kl_old": spear([p[0] for p in pts], [p[2] for p in pts]),
        "spearman_final_kl_new_plain_updates": spear(
            [x["forgetting"] for a, x in finals if a not in ("replay", "distill")],
            [x["kl_new"] for a, x in finals if a not in ("replay", "distill")]),
        "spearman_final_kl_new_without_replay": spear(
            [x["forgetting"] for a, x in finals if a != "replay"], [x["kl_new"] for a, x in finals if a != "replay"]),
    }
    L = max(out["arms"][a]["final"]["new_loss"]["mean"] for a in ARMS)
    matched = {}
    for a in ARMS:
        vals = []
        for rr in R[a]:
            xs = [row["new_loss"] for row in rr["log"]]
            ys = [row["old_loss"] - rr["log"][0]["old_loss"] for row in rr["log"]]
            vals.append(interp_at(xs, ys, L))
        matched[a] = stats([v for v in vals if v is not None]) | {"runs_reaching": sum(v is not None for v in vals)}
    out["matched"] = {"new_loss": r(L), "forgetting": matched}
    full = out["arms"]["full"]["curve_mean"]
    k_rep = out["arms"]["replay"]["final"]["kl_new"]["mean"]
    full_at = interp_at(full["kl_new"], full["forgetting"], k_rep)
    out["replay_vs_full_at_same_kl"] = {"kl_new": k_rep, "full_forgetting": None if full_at is None else r(full_at),
                                       "replay_forgetting": out["arms"]["replay"]["final"]["forgetting"]["mean"]}

    A = out["arms"]
    fin = {a: {s: R[a][s]["final"] for s in range(3)} for a in ARMS}
    E = {}
    E["E1"] = A["full"]["final"]["forgetting"]["mean"] >= 0.5, A["full"]["final"]["forgetting"]["mean"]
    E["E2"] = all(fin[lo][s]["new_loss"] > fin["full"][s]["new_loss"] and fin[lo][s]["forgetting"] < fin["full"][s]["forgetting"]
                  for lo in ("lora4", "lora32") for s in range(3)), {
        lo: {"new_loss": A[lo]["final"]["new_loss"]["mean"], "forgetting": A[lo]["final"]["forgetting"]["mean"]}
        for lo in ("full", "lora4", "lora32")}
    fr, ff = A["replay"]["final"]["forgetting"]["mean"], A["full"]["final"]["forgetting"]["mean"]
    dn = abs(A["replay"]["final"]["new_loss"]["mean"] - A["full"]["final"]["new_loss"]["mean"])
    E["E3"] = fr < ff / 3 and dn <= 0.1, {"replay_forgetting": fr, "full_forgetting": ff, "new_loss_gap": r(dn)}
    E["E4"] = A["distill"]["final"]["forgetting"]["mean"] < ff, A["distill"]["final"]["forgetting"]["mean"]
    on = out["one_number"]
    E["E5"] = on["spearman_final_kl_new"] >= 0.7, on["spearman_final_kl_new"]
    E["E6"] = on["spearman_final_kl_old"] > on["spearman_final_kl_new"], [on["spearman_final_kl_old"], on["spearman_final_kl_new"]]
    rv = out["replay_vs_full_at_same_kl"]
    E["E7"] = (rv["full_forgetting"] is not None and rv["replay_forgetting"] < rv["full_forgetting"] / 2), rv
    mm = {a: matched[a]["mean"] for a in ARMS}
    E["E8"] = min(mm, key=mm.get) == "replay", mm
    out["expectations_met"] = {k: {"met": bool(v[0]), "value": v[1], "text": exp["expectations"][k]} for k, v in E.items()}
    return out


# ---------------------------------------------------------------- section 4

def plasticity() -> dict:
    exp = read_results("expectations")
    out = {"arms": {}}
    for a in PLAST_ARMS:
        rs = [runs("plasticity", f"{a}_s{s}") for s in SEEDS]
        n = len(rs[0]["log"])
        acc = np.array([[row["online_acc"] for row in rr["log"]] for rr in rs])
        test = np.array([[row["test_acc"] for row in rr["log"]] for rr in rs])
        blocks = [r(acc[:, i:i + BLOCK].mean()) for i in range(0, n, BLOCK)]
        health = {}
        for k in ("dead", "erank", "weight"):
            rows = [[(row["task"], row[k]) for row in rr["log"] if k in row] for rr in rs]
            health[k] = {"tasks": [t for t, _ in rows[0]], "mean": [r(np.nanmean([rw[i][1] for rw in rows])) for i in range(len(rows[0]))],
                         "missing": sum(int(np.isnan(v)) for rw in rows for _, v in rw)}
        out["arms"][a] = {
            "spec": rs[0]["spec"], "tasks": n,
            "online_acc_blocks": blocks,
            "first_block": stats(acc[:, :BLOCK].mean(1)),
            "last_block": stats(acc[:, -BLOCK:].mean(1)),
            "drop": stats(acc[:, :BLOCK].mean(1) - acc[:, -BLOCK:].mean(1)),
            "test_first_block": r(test[:, :BLOCK].mean()), "test_last_block": r(test[:, -BLOCK:].mean()),
            "online_acc_mean_curve": [r(x) for x in acc.mean(0)],
            "health": health,
            "replaced": [rr["cbp"]["replaced"] for rr in rs] if rs[0]["cbp"] else None,
            "seconds": stats([rr["seconds"] for rr in rs]),
            "peak_rss_gb": max(rr["peak_rss_gb"] for rr in rs),
        }
    A = out["arms"]
    E = {}
    E["P1"] = A["backprop"]["drop"]["mean"] >= 0.03, A["backprop"]["drop"]["mean"]
    E["P2"] = A["cbp"]["drop"]["mean"] <= 0.01, A["cbp"]["drop"]["mean"]
    dead_bp, dead_cbp = A["backprop"]["health"]["dead"]["mean"][-1], A["cbp"]["health"]["dead"]["mean"][-1]
    E["P3"] = dead_bp > 0.5 and dead_cbp < 0.3, {"backprop": dead_bp, "cbp": dead_cbp}
    E["P4"] = A["sgd"]["drop"]["mean"] < A["backprop"]["drop"]["mean"], {"sgd": A["sgd"]["drop"]["mean"], "adam": A["backprop"]["drop"]["mean"]}
    fresh = r(np.mean(A["fresh"]["online_acc_mean_curve"]))
    E["P5"] = fresh - A["backprop"]["last_block"]["mean"] >= 0.03, {"fresh_mean": fresh, "backprop_last_block": A["backprop"]["last_block"]["mean"]}
    out["fresh_mean"] = fresh
    out["expectations_met"] = {k: {"met": bool(v[0]), "value": v[1], "text": exp["expectations"][k]} for k, v in E.items()}
    return out


# ---------------------------------------------------------------- section 5

def library() -> dict:
    exp = read_results("expectations")
    R = {a: runs("library", a) for a in LIB_ARMS}
    out = {"arms": {a: {k: R[a][k] for k in ("spec", "trainable_parameters", "log", "final", "seconds",
                                              "peak_mps_driver_gb", "peak_rss_gb")} for a in LIB_ARMS},
           "sets": R["lora8"]["sets"], "model": R["lora8"]["model"], "revision": R["lora8"]["revision"]}
    pts = [(row["probe_nll"] - R[a]["log"][0]["probe_nll"], row["kl_new"], row["kl_probe"], row["new_nll"])
           for a in LIB_ARMS for row in R[a]["log"][1:]]
    out["one_number"] = {"points": len(pts),
                         "spearman_kl_new": spear([p[0] for p in pts], [p[1] for p in pts]),
                         "spearman_kl_probe": spear([p[0] for p in pts], [p[2] for p in pts]),
                         "spearman_kl_new_without_replay": spear(
                             [p[0] for a in LIB_ARMS if a != "lora8_replay" for p in
                              [(row["probe_nll"] - R[a]["log"][0]["probe_nll"], row["kl_new"]) for row in R[a]["log"][1:]]],
                             [p[1] for a in LIB_ARMS if a != "lora8_replay" for p in
                              [(row["probe_nll"] - R[a]["log"][0]["probe_nll"], row["kl_new"]) for row in R[a]["log"][1:]]])}
    F = {a: R[a]["final"] for a in LIB_ARMS}
    E = {}
    lowered = all(F[a]["new_gain"] > 0 for a in LIB_ARMS)
    rose = all(F[a]["forgetting"] > 0 for a in ("lora8_gentle", "lora8", "lora64"))
    E["L1"] = lowered and rose and F["lora8"]["forgetting"] > F["lora8_gentle"]["forgetting"], {
        a: {"new_gain": F[a]["new_gain"], "forgetting": F[a]["forgetting"]} for a in LIB_ARMS}
    E["L2"] = out["one_number"]["spearman_kl_new"] >= 0.7, out["one_number"]["spearman_kl_new"]
    E["L3"] = F["lora64"]["forgetting"] > F["lora8"]["forgetting"], {"lora64": F["lora64"]["forgetting"], "lora8": F["lora8"]["forgetting"]}
    n8 = R["lora8"]["log"][-1]["new_nll"]; nr = R["lora8_replay"]["log"][-1]["new_nll"]
    E["L4"] = F["lora8_replay"]["forgetting"] < F["lora8"]["forgetting"] / 2 and abs(nr - n8) <= 0.05, {
        "replay_forgetting": F["lora8_replay"]["forgetting"], "lora8_forgetting": F["lora8"]["forgetting"],
        "new_nll_gap": r(abs(nr - n8))}
    out["expectations_met"] = {k: {"met": bool(v[0]), "value": v[1], "text": exp["expectations"][k]} for k, v in E.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--only", choices=["forget", "plasticity", "library"])
    args = ap.parse_args()
    for name, fn in (("forget", forget), ("plasticity", plasticity), ("library", library)):
        if args.only and args.only != name:
            continue
        res = fn()
        write_results(name, res)
        for k, v in res["expectations_met"].items():
            print(f"  {k}: {'met' if v['met'] else 'MISSED'}  {v['value']}")
    if args.figures:
        import figures
        figures.main()


if __name__ == "__main__":
    main()
