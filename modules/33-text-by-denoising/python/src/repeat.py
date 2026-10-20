"""Section 5: the data-repeat test, summarised from the repeat arms' run files.

Each repeat arm (see train.py) scored itself on the 256 held-out blocks and on
256 blocks of its own training characters at every epoch mark. This script
gathers them into one table per data size: the mean over seeds at each mark,
each arm's best mark, and the marks where the diffusion model's bound sits
below the AR twin's exact NLL.

    uv run python src/train.py --repeat      # the runs, in chunks (about 45 minutes on MPS)
    uv run python src/repeat.py              # results/repeat.json, from results/runs/r*.json
"""

from __future__ import annotations

import json

import common
import numpy as np

KEYS = {"ar": "nll", "ard3": "nll", "mdm": "elbo", "mdmlong": "elbo"}


def series(run: dict, split: str, key: str) -> dict:
    return {int(e): m[split][key] for e, m in run["marks"].items()}


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted((common.RESULTS / "runs").glob("r*.json"))]
    out = {"what": "held-out score per character (nats) at each epoch mark: AR exact NLL, diffusion ELBO and left-to-right exact score",
           "sizes": {}}
    for size in ("r100k", "r1m"):
        group = [r for r in runs if r["arm"].startswith(size + "_")]
        if not group:
            continue
        arms = {}
        for name, key in KEYS.items():
            rs = [r for r in group if r["arm"].split("_")[1] == name]
            if not rs:
                continue
            marks = sorted(series(rs[0], "val", key))
            val = np.array([[series(r, "val", key)[e] for e in marks] for r in rs])
            tr = np.array([[series(r, "train", key)[e] for e in marks] for r in rs])
            a = {"seeds": [r["spec"]["seed"] for r in rs], "dropout": rs[0]["spec"]["dropout"], "epochs": marks,
                 "val_mean": val.mean(0).round(4).tolist(), "val_min_seed": val.min(0).round(4).tolist(),
                 "val_max_seed": val.max(0).round(4).tolist(), "train_mean": tr.mean(0).round(4).tolist(),
                 "best_epoch": marks[int(val.mean(0).argmin())], "best_val": round(float(val.mean(0).min()), 4),
                 "last_minus_best": round(float(val.mean(0)[-1] - val.mean(0).min()), 4),
                 "minutes": [round(r["progress"]["seconds"] / 60, 2) for r in rs],
                 "chunks": [r["progress"]["chunks"] for r in rs], "steps": rs[0]["progress"]["step"]}
            if name.startswith("mdm"):
                l2r = np.array([[series(r, "val", "l2r")[e] for e in marks] for r in rs])
                a["val_l2r_mean"] = l2r.mean(0).round(4).tolist()
                a["best_l2r"] = round(float(l2r.mean(0).min()), 4)
            arms[name] = a
        if "ar" in arms and "mdm" in arms:
            ar, md = np.array(arms["ar"]["val_mean"]), np.array(arms["mdm"]["val_mean"])
            arms["crossover"] = {"epochs_mdm_elbo_below_ar": [e for e, x, y in zip(arms["ar"]["epochs"], md, ar) if x < y],
                                 "epochs_mdm_l2r_below_ar": [e for e, x, y in zip(arms["ar"]["epochs"],
                                                                                  arms["mdm"]["val_l2r_mean"], ar) if x < y],
                                 "best_gap_mdm_minus_ar": round(float(md.min() - ar.min()), 4)}
            if "mdmlong" in arms:
                lg = arms["mdmlong"]
                arms["crossover"]["mdmlong_epochs_below_ar_best"] = [e for e, x in zip(lg["epochs"], lg["val_mean"]) if x < ar.min()]
        out["sizes"][size] = {"unique_chars": group[0]["spec"]["U_chars"], **arms}
    common.write_results("repeat", out)


if __name__ == "__main__":
    main()
