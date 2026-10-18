"""Section 2: fit L(N) at 4M tokens and L(D) for the two largest sizes; a joint fit L(N, D).

    uv run python src/fit.py --figures

Reads results/train_d*.json (no model is trained here, CPU only) and writes
results/fit.json:

- a check of the fitter on synthetic losses with a known exponent;
- L(N): ordinary least squares of log L on log N over the six sizes, seed 0,
  at 4M tokens, and the same fit at 0.25, 0.5, 1 and 2M tokens;
- the noise band (the larger seed gap of the two smallest sizes) and each
  size step against it;
- L(D) for d 160 and 256: log L on log D over the marked points after warmup
  (0.5 to 4M tokens);
- a joint fit L(N, D) = E + A / N^alpha + B / D^beta over every seed-0 point
  after warmup, for the explorable and section 3;
- unigram and bigram baselines on the held-out text.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import (MARKED_STEPS, SIZES, STEPS, TWO_SEEDS, WARMUP, Guard, held_at, read_runs, tokens_at,
                    write_results)

AFTER_WARMUP = tuple(s for s in MARKED_STEPS if s > WARMUP)       # 122, 244, 488, 977


def fit_power(x, y) -> dict:
    """OLS of log y on log x. Returns the exponent (minus the slope), intercept a, R^2."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, a = np.polyfit(lx, ly, 1)
    pred = a + slope * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return {"exponent": float(-slope), "a": float(a), "r2": float(r2),
            "residuals_log": [round(float(r), 4) for r in ly - pred]}


def synthetic_check(ns, alpha: float = 0.1, noise: float = 0.01, draws: int = 1000) -> dict:
    """The fitter on log L = log 2 - alpha log N + noise, at the real N: does it give alpha back?"""
    def one(seed):
        rng = np.random.default_rng(seed)
        loss = np.exp(math.log(2.0) - alpha * np.log(ns) + rng.normal(0, noise, len(ns)))
        return fit_power(ns, loss)["exponent"]

    many = np.array([one(s) for s in range(1, draws + 1)])
    return {"true_alpha": alpha, "noise_sd_log": noise, "alpha_seed0": one(0),
            "draws": draws, "mean": float(many.mean()), "sd": float(many.std()),
            "min": float(many.min()), "max": float(many.max())}


def joint_fit(points: list[tuple[float, float, float]]) -> dict:
    """L = E + A N^-alpha + B D^-beta. For each (alpha, beta) on a grid, E, A, B by least squares
    on L; keep the best fit with E, A, B all >= 0."""
    n = np.array([p[0] for p in points])
    d = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    best = None
    for al in np.arange(0.02, 1.501, 0.01):
        xn = n ** -al
        for be in np.arange(0.02, 1.501, 0.01):
            X = np.stack([np.ones_like(y), xn, d ** -be], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if (coef < 0).any():
                continue
            sse = float(np.sum((X @ coef - y) ** 2))
            if best is None or sse < best[0]:
                best = (sse, al, be, coef)
    sse, al, be, (E, A, B) = best
    pred = E + A * n ** -al + B * d ** -be
    return {"form": "L = E + A / N^alpha + B / D^beta", "E": float(E), "A": float(A), "B": float(B),
            "alpha": round(float(al), 2), "beta": round(float(be), 2),
            "rmse": float(np.sqrt(sse / len(y))), "max_abs_err": float(np.abs(pred - y).max()),
            "n_points": len(y), "grid": "alpha, beta in 0.02..1.50 by 0.01"}


def baselines(train, held, vocab_size: int) -> dict:
    """Held-out nats per character of a unigram and a bigram table counted on the training part (add-one)."""
    u = np.bincount(train, minlength=vocab_size) + 1.0
    u /= u.sum()
    b = np.ones((vocab_size, vocab_size))
    np.add.at(b, (train[:-1], train[1:]), 1)
    b /= b.sum(1, keepdims=True)
    return {"unigram": float(-np.log(u[held]).mean()), "bigram": float(-np.log(b[held[:-1], held[1:]]).mean())}


def run() -> None:
    guard = Guard("fit.py")
    runs = read_runs()
    seed0 = {d: next(r for r in runs[d]["runs"] if r["seed"] == 0) for d in SIZES}
    ns = np.array([seed0[d]["n_non_embedding"] for d in SIZES], float)

    gaps = {}
    for d in TWO_SEEDS:
        a, b = (held_at(r, STEPS) for r in runs[d]["runs"])
        gaps[str(d)] = abs(a - b)
    band = max(gaps.values())
    loss4 = [held_at(seed0[d], STEPS) for d in SIZES]
    steps = [{"from": SIZES[i], "to": SIZES[i + 1], "drop": loss4[i] - loss4[i + 1]} for i in range(len(SIZES) - 1)]
    for s in steps:
        s["beyond_band"] = s["drop"] > band

    l_n = {str(st): {"tokens": tokens_at(st), **fit_power(ns, [held_at(seed0[d], st) for d in SIZES])}
           for st in MARKED_STEPS}
    seed1 = [held_at(next(r for r in runs[d]["runs"] if r["seed"] == 1), STEPS) if d in TWO_SEEDS
             else held_at(seed0[d], STEPS) for d in SIZES]
    l_n_seed1 = fit_power(ns, seed1)

    l_d = {}
    for d in SIZES:
        dd = [tokens_at(st) for st in AFTER_WARMUP]
        l_d[str(d)] = fit_power(dd, [held_at(seed0[d], st) for st in AFTER_WARMUP])

    pts = [(seed0[d]["n_non_embedding"], c["tokens"], c["held_loss"])
           for d in SIZES for c in seed0[d]["curve"] if c["step"] > WARMUP]
    joint = joint_fit(pts)
    guard.tick("fits")

    import corpus
    train, held, vocab = corpus.splits()
    base = baselines(train, held, vocab.size)
    guard.tick("baselines")

    out = {
        "sizes": list(SIZES), "n_non_embedding": ns.tolist(),
        "n_params_total": [seed0[d]["n_params_total"] for d in SIZES],
        "loss_4M_seed0": loss4, "seed_gaps_4M": gaps, "band": band, "size_steps": steps,
        "synthetic_check": synthetic_check(ns),
        "l_of_n": l_n, "l_of_n_main": l_n[str(STEPS)], "l_of_n_seed1_small": l_n_seed1,
        "l_of_d": l_d, "l_of_d_steps": list(AFTER_WARMUP), "l_of_d_largest_two": [SIZES[-2], SIZES[-1]],
        "joint": joint, "baselines": base,
        "guard": guard.report(),
    }
    write_results("fit", out)
    m = out["l_of_n_main"]
    print(f"  L(N) at 4M tokens: alpha {m['exponent']:.4f}, R2 {m['r2']:.3f}; band {band:.4f}")
    for d in SIZES[-2:]:
        print(f"  L(D) d {d}: beta {l_d[str(d)]['exponent']:.3f}, R2 {l_d[str(d)]['r2']:.3f}")
    print(f"  joint: {joint}")
    print(f"  baselines: {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="only redraw the figure")
    args = ap.parse_args()
    if not args.no_run:
        run()
    if args.figures:
        import figures
        figures.fig_fit()


if __name__ == "__main__":
    main()
