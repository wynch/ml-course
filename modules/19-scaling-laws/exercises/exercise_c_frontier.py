"""Exercise (c): the lowest loss at a compute budget, from the committed curves.

Each size logged its held-out loss at twelve points. With a constant learning
rate, the checkpoint at D tokens stands for a model trained on D tokens. Its
compute is C = 6 N D. For a budget C, find each size's loss there (interpolate
in log-log between checkpoints after warmup) and keep the best size.

Predict before you run: at C = 1e12 FLOPs, which d wins, a small model trained
long or a large one trained short? (d 32 at 4M tokens costs 1.2e12.)

    uv run python ../exercises/exercise_c_frontier.py
    uv run pytest tests/test_solutions.py -k frontier -v
"""

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
    # TODO(you): logs of both, return None if log c falls outside, else np.interp, then exp.
    raise NotImplementedError("TODO(you): write loss_at first")


def best_size(runs: dict[int, dict], c: float) -> tuple[int, float]:
    """(d_model, loss) of the size with the lowest loss at budget c, among those that reach it."""
    # TODO(you): loss_at for each size, drop the Nones, take the minimum.
    raise NotImplementedError("TODO(you): write best_size first")


def main() -> None:
    runs = load_seed0()
    for c in (3e11, 1e12, 3e12, 1e13, 3e13):
        d, loss = best_size(runs, c)
        n = runs[d]["n_non_embedding"]
        print(f"C {c:.0e}: best d {d} (N {n:,}), loss {loss:.3f}, D {c / (6 * n):,.0f} tokens, D/N {c / (6 * n * n):.1f}")


if __name__ == "__main__":
    main()
