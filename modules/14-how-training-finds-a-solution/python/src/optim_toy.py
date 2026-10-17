"""Section 1: SGD, momentum and Adam on two 2-D losses, written from scratch.

Each optimiser runs from a fixed start at 37 learning rates (10^-4 to 10^0.5, a
factor 10^0.125 apart) and we count the steps until the loss first drops below
1e-3, capped at 10,000. On these losses there are no batches, so "SGD" is
plain gradient descent: the noise that gives SGD its S is absent.

    uv run python src/optim_toy.py [--figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import Guard, write_results

TARGET = 1e-3
CAP = 10_000
LRS = [float(v) for v in 10 ** np.linspace(-4, 0.5, 37)]
MOMENTUM = 0.9
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

# The bowl is tilted 30 degrees, so its long axis lines up with neither
# coordinate; an axis-aligned bowl would favour Adam's per-coordinate scaling.
KAPPA = 25.0
ANGLE = np.pi / 6
_R = np.array([[np.cos(ANGLE), np.sin(ANGLE)], [-np.sin(ANGLE), np.cos(ANGLE)]])
_H = _R.T @ np.diag([1.0, KAPPA]) @ _R            # the bowl's Hessian, eigenvalues 1 and KAPPA


def bowl(p):
    return 0.5 * p @ _H @ p


def bowl_grad(p):
    return _H @ p


def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])


LOSSES = {
    "bowl": {"f": bowl, "grad": bowl_grad, "start": [-1.6, 1.4], "minimum": [0.0, 0.0],
             "box": [[-2.0, 2.0], [-1.6, 1.8]]},
    "rosenbrock": {"f": rosenbrock, "grad": rosenbrock_grad, "start": [-1.2, 1.0], "minimum": [1.0, 1.0],
                   "box": [[-2.0, 2.0], [-1.0, 3.0]]},
}


def sgd_step(p, g, state, lr):
    return p - lr * g


def momentum_step(p, g, state, lr):
    """Heavy ball, as torch.optim.SGD(momentum=0.9) writes it: v = mu v + g; p -= lr v."""
    v = MOMENTUM * state.get("v", np.zeros_like(p)) + g
    state["v"] = v
    return p - lr * v


def adam_step(p, g, state, lr):
    t = state.get("t", 0) + 1
    m = BETA1 * state.get("m", np.zeros_like(p)) + (1 - BETA1) * g
    v = BETA2 * state.get("v", np.zeros_like(p)) + (1 - BETA2) * g * g
    state.update(t=t, m=m, v=v)
    m_hat = m / (1 - BETA1 ** t)
    v_hat = v / (1 - BETA2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + EPS)


OPTIMISERS = {"sgd": sgd_step, "momentum": momentum_step, "adam": adam_step}


def run(loss: str, opt: str, lr: float, cap: int = CAP, keep_path: bool = False):
    """Steps until loss < TARGET (None if never within cap, or if it diverges)."""
    L = LOSSES[loss]
    p = np.array(L["start"], dtype=np.float64)
    state: dict = {}
    path = [p.tolist()]
    for step in range(1, cap + 1):
        p = OPTIMISERS[opt](p, L["grad"](p), state, lr)
        f = L["f"](p)
        if keep_path:
            path.append([round(float(v), 5) for v in p])
        if not np.isfinite(f) or f > 1e8:
            return {"steps": None, "diverged": True, "final_loss": None, "path": path}
        if f < TARGET:
            return {"steps": step, "diverged": False, "final_loss": float(f), "path": path}
    return {"steps": None, "diverged": False, "final_loss": float(f), "path": path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("optim_toy")

    out = {"target": TARGET, "cap": CAP, "lrs": LRS, "momentum": MOMENTUM,
           "adam": {"beta1": BETA1, "beta2": BETA2, "eps": EPS},
           "bowl": {"kappa": KAPPA, "angle_deg": 30}, "losses": {}}
    for loss, L in LOSSES.items():
        rec = {"start": L["start"], "minimum": L["minimum"], "box": L["box"],
               "start_loss": float(L["f"](np.array(L["start"]))), "optimisers": {}}
        for opt in OPTIMISERS:
            sweep = []
            for lr in LRS:
                r = run(loss, opt, lr)
                sweep.append({"lr": lr, "steps": r["steps"], "diverged": r["diverged"], "final_loss": r["final_loss"]})
            ok = [s for s in sweep if s["steps"] is not None]
            best = min(ok, key=lambda s: (s["steps"], s["lr"])) if ok else None
            best_path = run(loss, opt, best["lr"], keep_path=True)["path"] if best else []
            rec["optimisers"][opt] = {"sweep": sweep, "best_lr": best and best["lr"],
                                      "best_steps": best and best["steps"],
                                      "lrs_that_reach": len(ok), "best_path": best_path}
            print(f"{loss:10s} {opt:8s} best lr {best and best['lr']:.4g}  steps {best and best['steps']}"
                  f"  ({len(ok)} of {len(LRS)} rates reach {TARGET})")
        out["losses"][loss] = rec
    guard.tick("sweeps")
    out["guard"] = guard.report()
    write_results("optim_toy", out)
    if args.figures:
        import figures
        figures.fig_optim_toy()


if __name__ == "__main__":
    main()
