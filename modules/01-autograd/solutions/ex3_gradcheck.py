"""Solution 3 — grad_check implemented; it fingers the buggy op.

The bug lives in `buggy_engine.Value.__mul__`: its backward uses
`self.data` for BOTH parents,

    self.grad  += self.data * out.grad   # correct would be other.data
    other.grad += self.data * out.grad   # this line is right

so the derivative w.r.t. the first factor is wrong. For `a * 3.0` the analytic
grad comes out as `a` (=0.7) instead of the constant `3.0`, and the numeric
check flags it.

Run:  cd python && uv run ../solutions/ex3_gradcheck.py
"""

import sys
import pathlib

# import the shared buggy engine that ships with the exercises
_EX = pathlib.Path(__file__).resolve().parents[1] / "exercises"
sys.path.insert(0, str(_EX))
from buggy_engine import Value  # noqa: E402

ONE_LINE_ANSWER = "mul — its backward multiplies BOTH parents by self.data instead of the other factor"


def grad_check(build, x0, eps=1e-6):
    """Return (analytic_grad, numeric_grad) of `build` at scalar input x0."""
    # analytic
    a = Value(x0)
    out = build(a)
    out.backward()
    analytic = a.grad
    # numeric, central difference on the forward *data*
    numeric = (build(Value(x0 + eps)).data - build(Value(x0 - eps)).data) / (2 * eps)
    return analytic, numeric


def main():
    eps_tol = 1e-4
    probes = {
        "add": lambda a: a + 3.0,
        "mul": lambda a: a * 3.0,
        "pow": lambda a: a ** 3,
        "exp": lambda a: a.exp(),
        "tanh": lambda a: a.tanh(),
        "relu": lambda a: a.relu(),
    }
    x0 = 0.7

    print(f"{'op':6s} {'analytic':>12s} {'numeric':>12s} {'err':>10s}  status")
    guilty = []
    for name, fn in probes.items():
        ga, gn = grad_check(fn, x0)
        err = abs(ga - gn)
        status = "ok" if err < eps_tol else "  <-- BUG"
        if err >= eps_tol:
            guilty.append(name)
        print(f"{name:6s} {ga:12.6f} {gn:12.6f} {err:10.1e}  {status}")

    print(f"\nbuggy op(s): {guilty}")
    print(f"answer: {ONE_LINE_ANSWER}")
    assert guilty == ["mul"], "expected exactly the mul op to fail the check"


if __name__ == "__main__":
    main()
