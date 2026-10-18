"""Section 3: at each compute budget C = 6 N D, which size gives the lowest held-out loss?

    uv run python src/frontier.py --figures

Reads results/train_d*.json and results/fit.json (CPU only) and writes
results/frontier.json.

Every checkpoint after warmup is a run of D tokens at a constant learning rate,
so it stands for a model trained on exactly that many tokens. For each size we
interpolate its loss linearly in (log C, log L) between checkpoints, then, at
40 budgets spaced evenly in log C, keep the size with the lowest loss. N is the
non-embedding count, as in section 2.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import SIZES, TWO_SEEDS, WARMUP, Guard, read_results, read_runs, write_results

N_BUDGETS = 40


def curve_in_c(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """(log C, log L) for one run's checkpoints after warmup."""
    pts = [(6.0 * run["n_non_embedding"] * c["tokens"], c["held_loss"]) for c in run["curve"] if c["step"] > WARMUP]
    c, l = np.array(pts).T
    return np.log(c), np.log(l)


def loss_at(run: dict, c: float) -> float | None:
    lc, ll = curve_in_c(run)
    x = np.log(c)
    if x < lc[0] - 1e-9 or x > lc[-1] + 1e-9:
        return None
    return float(np.exp(np.interp(x, lc, ll)))


def frontier(runs_by_size: dict[int, dict], budgets: np.ndarray) -> list[dict]:
    rows = []
    for c in budgets:
        cands = {d: loss_at(r, c) for d, r in runs_by_size.items()}
        cands = {d: v for d, v in cands.items() if v is not None}
        d_best = min(cands, key=cands.get)
        n_best = runs_by_size[d_best]["n_non_embedding"]
        rows.append({"C": float(c), "best_d": d_best, "best_n": n_best, "loss": cands[d_best],
                     "tokens": float(c / (6 * n_best)), "tokens_per_param": float(c / (6 * n_best * n_best)),
                     "candidates": {str(d): v for d, v in cands.items()}})
    return rows


def run() -> None:
    guard = Guard("frontier.py")
    runs = read_runs()
    seed0 = {d: next(r for r in runs[d]["runs"] if r["seed"] == 0) for d in SIZES}
    seed1 = {d: next((r for r in runs[d]["runs"] if r["seed"] == 1), seed0[d]) for d in SIZES}

    lo = min(np.exp(curve_in_c(r)[0][0]) for r in seed0.values())
    hi = max(np.exp(curve_in_c(r)[0][-1]) for r in seed0.values())
    budgets = np.geomspace(lo, hi, N_BUDGETS)
    rows = frontier(seed0, budgets)
    rows1 = frontier(seed1, budgets)
    same = sum(a["best_d"] == b["best_d"] for a, b in zip(rows, rows1))

    winners = []                                   # the budgets where each size is best
    for r in rows:
        if not winners or winners[-1]["d"] != r["best_d"]:
            winners.append({"d": r["best_d"], "n": r["best_n"], "from_C": r["C"], "to_C": r["C"]})
        winners[-1]["to_C"] = r["C"]

    # N_opt against C: a line through the budgets where a size other than the
    # largest wins (where d 256 wins, nothing larger was trained to compete).
    inner = [r for r in rows if r["best_d"] != SIZES[-1]]
    slope = None
    if len({r["best_n"] for r in inner}) > 1:
        slope = float(np.polyfit(np.log([r["C"] for r in inner]), np.log([r["best_n"] for r in inner]), 1)[0])

    j = read_results("fit")["joint"]
    a_exp = j["beta"] / (j["alpha"] + j["beta"])   # N_opt ∝ C^(beta / (alpha + beta)) for the joint form

    write_results("frontier", {
        "budgets": budgets.tolist(), "rows": rows, "rows_seed1_small": [r["best_d"] for r in rows1],
        "same_winner_with_seed1": same, "winners": winners,
        "n_opt_slope_steps": slope, "n_opt_slope_points": len(inner),
        "joint_n_opt_exponent": a_exp,
        "flops_note": {
            "attention_share_of_forward_flops": {
                str(d): round(2 * 4 * 128 * d / (2 * seed0[d]["n_non_embedding"] + 2 * 4 * 128 * d), 3) for d in SIZES},
            "rule": "forward per token about 2N + 2 n_layer n_ctx d_model; 6ND keeps only the 6N part",
        },
        "guard": guard.report(),
    })
    print(f"  winners: {[(w['d'], f'{w['from_C']:.2e}', f'{w['to_C']:.2e}') for w in winners]}")
    print(f"  N_opt slope over steps: {slope}; joint-fit exponent {a_exp:.3f}; same winner with seed 1: {same}/{N_BUDGETS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="only redraw the figure")
    args = ap.parse_args()
    if not args.no_run:
        run()
    if args.figures:
        import figures
        figures.fig_frontier()


if __name__ == "__main__":
    main()
