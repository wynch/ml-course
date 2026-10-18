"""Sections 1 and 2: train a 2-layer transformer to regress in context, then score it
against least squares, ridge, 3-NN and one gradient step, by number of pairs.

    uv run python src/train_icl.py [--figures]

Writes results/icl.json, artifacts/icl_net.pt and artifacts/icl_control.pt.

The recipe, the baselines, the three bars and the control below were written down
before the first run:
- net MSE/d at 10 pairs <= 0.05; at 5 pairs within 1.5x ridge (lambda 0.1); at 20 pairs <= 0.02;
- known answer: least squares MSE < 1e-6 at every n >= d (the task has no noise);
- control: a second net, same recipe but 10,000 steps, trained and scored on prompts whose
  y come from another prompt; its mean MSE/d over n = 0..20 must lie in [0.8, 1.2].
"""

from __future__ import annotations

import argparse

from common import ARTIFACTS, Guard, free, get_device, write_results

import numpy as np
import torch

import icl

BARS = {"n10_max": 0.05, "n5_ratio_to_ridge_max": 1.5, "n20_max": 0.02}
CONTROL_BAND = (0.8, 1.2)
KNOWN_ANSWER_MAX = 1e-6
SEED_ORDER = 4                 # numpy seed for the pair-order permutation check


def order_check(net, xs, ys, device) -> dict:
    """Shuffle the order of the 20 context pairs (x and y together), keep the query last."""
    rng = np.random.default_rng(SEED_ORDER)
    xs_p, ys_p = xs.copy(), ys.copy()
    last = xs.shape[1] - 1
    for b in range(len(xs)):
        perm = rng.permutation(last)
        xs_p[b, :last], ys_p[b, :last] = xs[b, perm], ys[b, perm]
    p0 = icl.net_predict(net, xs, ys, device)[:, last]
    p1 = icl.net_predict(net, xs_p, ys_p, device)[:, last]
    yq = ys[:, last]
    return {"n": last,
            "mse_over_d_original_order": float(((p0 - yq) ** 2).mean() / icl.D),
            "mse_over_d_shuffled_order": float(((p1 - yq) ** 2).mean() / icl.D),
            "mean_sq_change_over_d": float(((p1 - p0) ** 2).mean() / icl.D)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("train_icl")
    device = get_device()

    xs, ys, _ = icl.eval_set(icl.SEED_EVAL, shuffled=False)
    xs_s, ys_s, _ = icl.eval_set(icl.SEED_EVAL_SHUFFLED, shuffled=True)
    err = icl.baselines(xs, ys)
    base = {k: (v.mean(0) / icl.D).tolist() for k, v in err.items()}
    known = {str(n): float(err["least_squares"][:, n].mean()) for n in range(icl.D, icl.N_POINTS)}
    guard.tick("baselines")

    print("training the main net")
    log_main: list = []
    torch.manual_seed(icl.SEED_WEIGHTS)
    net = icl.train(icl.ICLNet(), icl.STEPS, shuffled=False, device=device, log=log_main)
    guard.tick("train main")
    mse_net = icl.mse_over_d(icl.net_predict(net, xs, ys, device), ys)
    mse_net_shuf = icl.mse_over_d(icl.net_predict(net, xs_s, ys_s, device), ys_s)
    order = order_check(net, xs, ys, device)

    # The same forward on the CPU: how far MPS float32 drifts.
    cpu_net = icl.ICLNet()
    cpu_net.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    cpu_net.eval()
    p_dev = icl.net_predict(net, xs[:250], ys[:250], device)
    p_cpu = icl.net_predict(cpu_net, xs[:250], ys[:250], torch.device("cpu"))
    mps_vs_cpu = {"device": device.type, "prompts": 250, "max_abs_diff": float(np.abs(p_dev - p_cpu).max()),
                  "prediction_range": [float(p_cpu.min()), float(p_cpu.max())]}
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(cpu_net.state_dict(), ARTIFACTS / "icl_net.pt")
    del net, cpu_net
    free()

    print("training the shuffled-y control net")
    log_ctrl: list = []
    torch.manual_seed(icl.SEED_WEIGHTS)
    ctrl = icl.train(icl.ICLNet(), icl.CONTROL_STEPS, shuffled=True, device=device, log=log_ctrl)
    guard.tick("train control")
    mse_ctrl = icl.mse_over_d(icl.net_predict(ctrl, xs_s, ys_s, device), ys_s)
    torch.save({k: v.cpu() for k, v in ctrl.state_dict().items()}, ARTIFACTS / "icl_control.pt")
    del ctrl
    free()

    ridge5 = base["ridge_0.1"][5]
    bars = {
        "known_answer": {"value": max(known.values()), "bar": f"< {KNOWN_ANSWER_MAX}",
                         "met": max(known.values()) < KNOWN_ANSWER_MAX},
        "control": {"value": float(mse_ctrl.mean()), "bar": list(CONTROL_BAND),
                    "met": bool(CONTROL_BAND[0] <= mse_ctrl.mean() <= CONTROL_BAND[1])},
        "n10": {"value": float(mse_net[10]), "bar": f"<= {BARS['n10_max']}", "met": bool(mse_net[10] <= BARS["n10_max"])},
        "n5": {"value": float(mse_net[5]), "ridge": ridge5, "ratio": float(mse_net[5] / ridge5),
               "bar": f"<= {BARS['n5_ratio_to_ridge_max']}x ridge", "met": bool(mse_net[5] <= 1.5 * ridge5)},
        "n20": {"value": float(mse_net[20]), "bar": f"<= {BARS['n20_max']}", "met": bool(mse_net[20] <= BARS["n20_max"])},
        "falls_5_to_10": bool(mse_net[10] < mse_net[5]),
    }
    for k, v in bars.items():
        print(f"  {k}: {v}")

    payload = {
        "config": {"d": icl.D, "n_points": icl.N_POINTS, "d_model": 128, "heads": 4, "d_mlp": 512, "layers": 2,
                   "params": sum(p.numel() for p in icl.ICLNet().parameters()), "steps": icl.STEPS,
                   "control_steps": icl.CONTROL_STEPS, "batch": icl.BATCH, "lr": icl.LR, "lr_min": icl.LR_MIN,
                   "warmup": icl.WARMUP, "clip": icl.CLIP, "n_eval": icl.N_EVAL, "ridge_lambda": icl.RIDGE_LAMBDA,
                   "knn_k": icl.KNN_K, "device": device.type},
        "mse_over_d": {"n": list(range(icl.N_POINTS)), "net": mse_net.tolist(), **base,
                       "control_net_on_shuffled": mse_ctrl.tolist(), "main_net_on_shuffled": mse_net_shuf.tolist()},
        "known_answer_ls_mse": known,
        "bars": bars,
        "order_permutation": order,
        "mps_vs_cpu": mps_vs_cpu,
        "train_loss_every_100": {"main": log_main, "control": log_ctrl},
        "guard": guard.report(),
    }
    write_results("icl", payload)
    if args.figures:
        import figures
        figures.fig_icl()


if __name__ == "__main__":
    main()
