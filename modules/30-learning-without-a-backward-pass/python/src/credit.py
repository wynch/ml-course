"""Section 3: how predictive coding's error reaches the lower layers.

    uv run python src/credit.py            # results/credit.json
    uv run python src/credit.py --figures  # the same, then redraws figures/credit.png

Two networks at their initial weights (seed 0): section 2's (three hidden
layers of 256) and a deeper one (eight hidden layers of 256). On 256 test
images, backprop's gradient for each weight matrix is the reference. Then
predictive coding's inference runs for T = 0 to 200 steps and, after each,
we read the weight update it would make: its cosine with backprop's gradient
and its size relative to it, layer by layer. CPU, seconds.
"""

from __future__ import annotations

import sys

import common
import torch

import rules as R

STEPS = 200
DEPTHS = {"3 hidden": (784, 256, 256, 256, 10), "8 hidden": (784,) + (256,) * 8 + (10,)}


def compare(sizes, x, y, gamma: float = 0.1) -> dict:
    bp = R.Backprop(sizes, 0)
    loss = torch.nn.functional.cross_entropy(bp.forward(x), y)
    ref = [g.detach() for g in torch.autograd.grad(loss, bp.params)][0::2]          # weights only
    pc = R.PredictiveCoding(sizes, 0, gamma=gamma)
    trace = [pc.weight_grads(pc.feedforward(x)[:-1], y)]                            # T = 0
    pc.infer(x, y, T=STEPS, trace=trace)
    L = len(ref)
    cos = [[None] * (STEPS + 1) for _ in range(L)]
    size = [[None] * (STEPS + 1) for _ in range(L)]
    for t, g in enumerate(trace):
        for l, (u, v) in enumerate(zip(g[0::2], ref)):
            n = float(u.norm())
            size[l][t] = float(f"{n / float(v.norm()):.4g}")
            cos[l][t] = None if n == 0 else round(float(torch.nn.functional.cosine_similarity(u.flatten(), v.flatten(), 0)), 4)
    first = [next(t for t in range(STEPS + 1) if cos[l][t] is not None) for l in range(L)]
    good = [next((t for t in range(STEPS + 1) if cos[l][t] is not None and cos[l][t] > 0.95), None) for l in range(L)]
    return {"sizes": list(sizes), "gamma": gamma, "cosine": cos, "size_ratio": size,
            "first_nonzero_step": first,
            "cosine_on_arrival": [cos[l][first[l]] for l in range(L)],
            "first_step_cosine_above_0_95": good,
            "at_T": {str(T): {"cosine": [cos[l][T] for l in range(L)], "size_ratio": [size[l][T] for l in range(L)]}
                     for T in (1, 5, 10, 20, 50, 200)}}


def main() -> dict:
    guard = common.Guard("credit")
    _, _, x_te, y_te = common.load_fashion()
    x, y = x_te[:256], y_te[:256]
    out = {"images": "the first 256 test images", "steps": STEPS,
           "layers": "index 0 is the lowest weight matrix (input to first hidden), the last is the output layer",
           "nets": {k: compare(s, x, y) for k, s in DEPTHS.items()}}
    for k, r in out["nets"].items():
        print(k, "first non-zero step per layer:", r["first_nonzero_step"])
        print("  cosine on arrival:", r["cosine_on_arrival"])
        print("  T=20 size ratio:", r["at_T"]["20"]["size_ratio"])
    out["guard"] = guard.report()
    common.write_results("credit", out)
    return out


if __name__ == "__main__":
    main()
    if "--figures" in sys.argv:
        import figures
        figures.credit()
