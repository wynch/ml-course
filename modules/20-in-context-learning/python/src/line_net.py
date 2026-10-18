"""The explorable's net: the same recipe on lines in the plane, small enough to run in a browser.

A task is a line y = a u + b with a, b ~ N(0, 1); points have u ~ N(0, 1). The net sees
x = (u, 1), so a line is a 2-dimensional linear task and least squares on (u, 1) is the
ordinary fitted line. d_model 32, 4 heads, MLP 128, 2 blocks, 10,000 steps on the CPU.

    uv run python src/line_net.py [--figures]

Writes results/line.json and artifacts/line_net.pt; export_explorable.py copies the
weights into the page.
"""

from __future__ import annotations

import argparse

from common import ARTIFACTS, Guard, write_results

import numpy as np
import torch

import icl

D_LINE = 2
STEPS = 10_000
SHAPE = {"d_model": 32, "heads": 4, "d_mlp": 128, "layers": 2}
SEED_EVAL = 5
DEMO_PROMPT = {"u": [-1.6, -0.7, 0.3, 1.1, 1.9], "y": [-1.9, -0.6, 0.2, 1.4, 2.3]}
GRID = [round(-3 + 0.25 * i, 2) for i in range(25)]


def make_net() -> icl.ICLNet:
    return icl.ICLNet(d=D_LINE, **SHAPE)


def line_xs(n_prompts: int, gen: torch.Generator):
    w = torch.randn(n_prompts, D_LINE, generator=gen)
    u = torch.randn(n_prompts, icl.N_POINTS, generator=gen)
    xs = torch.stack([u, torch.ones_like(u)], -1)
    return xs, torch.einsum("bnd,bd->bn", xs, w)


def eval_lines(seed: int = SEED_EVAL, n: int = icl.N_EVAL):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, D_LINE))
    u = rng.standard_normal((n, icl.N_POINTS))
    xs = np.stack([u, np.ones_like(u)], -1)
    return xs, np.einsum("bnd,bd->bn", xs, w)


@torch.no_grad()
def predict_grid(net: icl.ICLNet, u_ctx: list[float], y_ctx: list[float], grid: list[float]) -> list[float]:
    """The net's guess at each query u, given the context points."""
    n = len(u_ctx)
    xs = torch.tensor([[[u, 1.0] for u in u_ctx] + [[q, 1.0]] for q in grid])
    ys = torch.tensor([y_ctx + [0.0] for _ in grid])
    return net(xs, ys)[:, n].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("line_net")
    torch.manual_seed(icl.SEED_WEIGHTS)
    log: list = []
    net = icl.train(make_net(), STEPS, shuffled=False, device=torch.device("cpu"), log=log, xs_fn=line_xs)
    guard.tick("train")

    xs, ys = eval_lines()
    mse_net = icl.mse_over_d(icl.net_predict(net, xs, ys, torch.device("cpu")), ys, D_LINE)
    err = icl.baselines(xs, ys)
    base = {k: (err[k].mean(0) / D_LINE).tolist() for k in ("least_squares", "ridge_0.1")}

    X = np.stack([DEMO_PROMPT["u"], np.ones(5)], -1)
    a, b = icl.w_least_squares(X, np.array(DEMO_PROMPT["y"]))
    demo = {**DEMO_PROMPT, "grid": GRID, "net": predict_grid(net, DEMO_PROMPT["u"], DEMO_PROMPT["y"], GRID),
            "least_squares": [float(a * q + b) for q in GRID], "ls_slope_intercept": [float(a), float(b)]}

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), ARTIFACTS / "line_net.pt")
    payload = {
        "config": {"task": "y = a u + b, a, b ~ N(0, 1), u ~ N(0, 1); x = (u, 1)", "d": D_LINE, **SHAPE,
                   "params": sum(p.numel() for p in net.parameters()), "steps": STEPS, "batch": icl.BATCH,
                   "lr": icl.LR, "device": "cpu", "n_eval": icl.N_EVAL},
        "mse_over_d": {"n": list(range(icl.N_POINTS)), "net": mse_net.tolist(), **base},
        "demo": demo,
        "train_loss_every_100": log,
        "guard": guard.report(),
    }
    write_results("line", payload)
    print("  MSE/d at n = 2, 5, 10, 20:", [round(float(mse_net[n]), 4) for n in (2, 5, 10, 20)])
    if args.figures:
        import figures
        figures.fig_line()


if __name__ == "__main__":
    main()
