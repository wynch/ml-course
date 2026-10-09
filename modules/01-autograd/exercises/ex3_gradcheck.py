"""Exercise 3 — write a numerical gradient checker and find the buggy op.

`buggy_engine.py` next to this file is a full autograd engine with exactly ONE
incorrect backward pass. Write `grad_check` below, then run the harness — it
probes each operation in isolation and prints which one disagrees with the
finite-difference gradient. Name the guilty op in ONE_LINE_ANSWER.

Run:  cd python && uv run ../exercises/ex3_gradcheck.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from buggy_engine import Value  # noqa: E402

ONE_LINE_ANSWER = "TODO(you): which op is buggy? e.g. 'mul' / 'pow' / 'tanh' ..."


def grad_check(build, x0, eps=1e-6):
    """Return (analytic_grad, numeric_grad) of `build` at scalar input x0.

    `build` takes a single Value and returns a single output Value.

    TODO(you):
      analytic: create a = Value(x0), out = build(a), call out.backward(),
                read a.grad.
      numeric : central difference of the *data* of build(Value(x0 +/- eps)).
    """
    raise NotImplementedError("implement grad_check")


def main():
    eps_tol = 1e-4
    # one probe per op; each is a function of a single input `a`
    probes = {
        "add": lambda a: a + 3.0,
        "mul": lambda a: a * 3.0,
        "pow": lambda a: a ** 3,
        "exp": lambda a: a.exp(),
        "tanh": lambda a: a.tanh(),
        "relu": lambda a: a.relu(),
    }
    x0 = 0.7  # positive so relu is in its linear regime

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
    print(f"your answer: {ONE_LINE_ANSWER}")


if __name__ == "__main__":
    main()
