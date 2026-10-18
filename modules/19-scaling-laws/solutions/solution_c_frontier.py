"""Reference solution, exercise (c): the lowest loss at a compute budget, from the committed curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIZES = (32, 48, 64, 96, 160, 256)
WARMUP = 100


def load_seed0() -> dict[int, dict]:
    out = {}
    for d in SIZES:
        r = json.loads((RESULTS / f"train_d{d}.json").read_text())
        out[d] = next(x for x in r["runs"] if x["seed"] == 0)
    return out


def loss_at(run: dict, c: float) -> float | None:
    """Held-out loss of this run at compute c = 6 N D, interpolated linearly in (log C, log L)
    between its checkpoints after warmup; None outside the range they cover."""
    n = run["n_non_embedding"]
    pts = [(6.0 * n * p["tokens"], p["held_loss"]) for p in run["curve"] if p["step"] > WARMUP]
    lc = np.log([p[0] for p in pts])
    ll = np.log([p[1] for p in pts])
    x = np.log(c)
    if x < lc[0] - 1e-9 or x > lc[-1] + 1e-9:
        return None
    return float(np.exp(np.interp(x, lc, ll)))


def best_size(runs: dict[int, dict], c: float) -> tuple[int, float]:
    """(d_model, loss) of the size with the lowest loss at budget c, among those that reach it."""
    cands = {d: loss_at(r, c) for d, r in runs.items()}
    cands = {d: v for d, v in cands.items() if v is not None}
    d = min(cands, key=cands.get)
    return d, cands[d]


def main() -> None:
    runs = load_seed0()
    for c in (3e11, 1e12, 3e12, 1e13, 3e13):
        d, loss = best_size(runs, c)
        n = runs[d]["n_non_embedding"]
        print(f"C {c:.0e}: best d {d} (N {n:,}), loss {loss:.3f}, D {c / (6 * n):,.0f} tokens, D/N {c / (6 * n * n):.1f}")


if __name__ == "__main__":
    main()
