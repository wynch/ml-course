"""Reference solution, exercise (a): momentum and Adam, one step each, in NumPy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import numpy as np

from optim_toy import CAP, TARGET, rosenbrock, rosenbrock_grad


def momentum_step(p, g, state, lr, mu=0.9):
    """Heavy ball: v = mu v + g; p = p - lr v. ``state`` keeps v between calls."""
    v = mu * state.get("v", np.zeros_like(p)) + g
    state["v"] = v
    return p - lr * v


def adam_step(p, g, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam: running means of g and g squared, bias-corrected, then a per-coordinate step."""
    t = state.get("t", 0) + 1
    m = b1 * state.get("m", np.zeros_like(p)) + (1 - b1) * g
    v = b2 * state.get("v", np.zeros_like(p)) + (1 - b2) * g * g
    state.update(t=t, m=m, v=v)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)


def steps_to_target(step_fn, lr, start=(-1.2, 1.0), cap=CAP):
    p, state = np.array(start, dtype=np.float64), {}
    for k in range(1, cap + 1):
        p = step_fn(p, rosenbrock_grad(p), state, lr)
        f = rosenbrock(p)
        if not np.isfinite(f) or f > 1e8:
            return None
        if f < TARGET:
            return k
    return None


def main() -> None:
    for name, fn, lr in (("momentum", momentum_step, 0.0031623), ("adam", adam_step, 1.3335214)):
        print(f"{name:8s} lr {lr:<10g} steps to Rosenbrock loss < 1e-3: {steps_to_target(fn, lr)}")


if __name__ == "__main__":
    main()
