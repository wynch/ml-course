"""Section 3, a demo: does the trained net look like least squares or like one gradient step?

Two read-outs of the net from section 1 (artifacts/icl_net.pt), on the CPU:

1. Implied weights. The net's guess for a query is some function of the query x. Its
   gradient with respect to x, at the query, is the weight vector the net acts as if it
   had fitted. We compare it, by cosine, with least squares' w and with one gradient
   step's w (X^T y), at each n.
2. Probes. At 10 pairs, a ridge probe reads least squares' w (here the true w) and one
   gradient step's w from the residual stream at the query token: after the embedding,
   after block 1, after block 2. Fit on 4,000 prompts, scored (R^2) on 1,000 others. The
   same probes on the untrained net are the control.

    uv run python src/gd_demo.py [--figures]
"""

from __future__ import annotations

import argparse

from common import ARTIFACTS, Guard, write_results

import numpy as np
import torch

import icl

N_IMPLIED = 500
SEED_IMPLIED, SEED_PROBE_FIT, SEED_PROBE_TEST = 6, 7, 8
PROBE_N = 10
PROBE_FIT, PROBE_TEST = 4000, 1000
PROBE_LAMBDA = 1.0


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def implied_weights(net: icl.ICLNet) -> dict:
    xs, ys, _ = icl.eval_set(SEED_IMPLIED, shuffled=False, n=N_IMPLIED)
    out = {"n": [], "cos_net_ls": [], "cos_net_gd": [], "cos_gd_ls": [], "norm_ratio_net_ls": []}
    for n in range(1, icl.N_POINTS):
        x = torch.from_numpy(xs[:, : n + 1]).float().requires_grad_(True)
        y = torch.from_numpy(ys[:, : n + 1]).float()
        pred = net(x, y)[:, n]
        (g,) = torch.autograd.grad(pred.sum(), x)
        w_net = g[:, n].double().numpy()             # d(prediction)/d(query x); earlier x feed other rows only
        w_ls = np.stack([icl.w_least_squares(xs[b, :n], ys[b, :n]) for b in range(N_IMPLIED)])
        w_gd = np.stack([icl.w_one_gd_step(xs[b, :n], ys[b, :n]) for b in range(N_IMPLIED)])
        out["n"].append(n)
        out["cos_net_ls"].append(float(cos(w_net, w_ls).mean()))
        out["cos_net_gd"].append(float(cos(w_net, w_gd).mean()))
        out["cos_gd_ls"].append(float(cos(w_gd, w_ls).mean()))
        out["norm_ratio_net_ls"].append(float(np.median(np.linalg.norm(w_net, axis=1) /
                                                        (np.linalg.norm(w_ls, axis=1) + 1e-12))))
    return out


@torch.no_grad()
def residuals(net: icl.ICLNet, seed: int, n_prompts: int):
    xs, ys, _ = icl.eval_set(seed, shuffled=False, n=n_prompts)
    xs, ys = xs[:, : PROBE_N + 1], ys[:, : PROBE_N + 1]
    _, hs = net(torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), hidden=True)
    feats = [h[:, PROBE_N].double().numpy() for h in hs]
    w_ls = np.stack([icl.w_least_squares(xs[b, :PROBE_N], ys[b, :PROBE_N]) for b in range(n_prompts)])
    w_gd = np.stack([icl.w_one_gd_step(xs[b, :PROBE_N], ys[b, :PROBE_N]) for b in range(n_prompts)])
    return feats, {"least_squares": w_ls, "one_gd_step": w_gd}


def ridge_r2(f_fit, t_fit, f_test, t_test) -> float:
    mu, sd = f_fit.mean(0), f_fit.std(0) + 1e-8
    a, b = (f_fit - mu) / sd, (f_test - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    coef = np.linalg.solve(a.T @ a + PROBE_LAMBDA * np.eye(a.shape[1]), a.T @ t_fit)
    resid = ((b @ coef - t_test) ** 2).sum()
    total = ((t_test - t_test.mean(0)) ** 2).sum()
    return float(1 - resid / total)


def probes(net: icl.ICLNet) -> dict:
    f_fit, t_fit = residuals(net, SEED_PROBE_FIT, PROBE_FIT)
    f_test, t_test = residuals(net, SEED_PROBE_TEST, PROBE_TEST)
    return {target: [ridge_r2(f_fit[i], t_fit[target], f_test[i], t_test[target]) for i in range(len(f_fit))]
            for target in t_fit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("gd_demo")
    torch.set_grad_enabled(True)
    net = icl.ICLNet()
    net.load_state_dict(torch.load(ARTIFACTS / "icl_net.pt"))
    net.eval()
    torch.manual_seed(icl.SEED_WEIGHTS)
    untrained = icl.ICLNet().eval()

    implied = implied_weights(net)
    guard.tick("implied weights")
    pr = {"trained": probes(net), "untrained": probes(untrained)}
    guard.tick("probes")
    for i in (4, 9, 19):
        print(f"  n {implied['n'][i]}: cos(net, LS) {implied['cos_net_ls'][i]:.3f}  cos(net, GD) "
              f"{implied['cos_net_gd'][i]:.3f}  cos(GD, LS) {implied['cos_gd_ls'][i]:.3f}")
    print("  probes R^2 by layer:", pr)

    write_results("gd_demo", {
        "config": {"n_implied": N_IMPLIED, "probe_n": PROBE_N, "probe_fit": PROBE_FIT, "probe_test": PROBE_TEST,
                   "probe_lambda": PROBE_LAMBDA, "layers": ["embedding", "block 1", "block 2"], "device": "cpu"},
        "implied": implied,
        "probes_r2": pr,
        "guard": guard.report(),
    })
    if args.figures:
        import figures
        figures.fig_gd()


if __name__ == "__main__":
    main()
