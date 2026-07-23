"""Numerical gradient checking: compare analytic grads (from backward) against
finite-difference estimates, and plot them side by side.

For each parameter p we compare:
  analytic  = p.grad   (from backward())
  numeric   = (L(p+eps) - L(p-eps)) / (2*eps)

If the engine is correct the two bars match to ~1e-6. Run:
  uv run scripts/grad_check.py
"""

import _bootstrap  # noqa: F401
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from micrograd.engine import Value


def build_expr(vals):
    """A scalar function of several inputs, exercising +, *, pow, tanh, exp, relu.

    f = tanh(a*b + c**2) + relu(d) * exp(e) - a/c
    Returns (output_value, list_of_input_Value_nodes).
    """
    a, b, c, d, e = [Value(v) for v in vals]
    out = (a * b + c**2).tanh() + d.relu() * e.exp() - a / c
    return out, [a, b, c, d, e]


def loss_from(vals):
    out, _ = build_expr(vals)
    return out.data


def main() -> None:
    names = ["a", "b", "c", "d", "e"]
    vals = [1.5, -2.0, 3.0, 0.7, 0.4]

    # analytic gradients via backward
    out, inputs = build_expr(vals)
    out.backward()
    analytic = np.array([p.grad for p in inputs])

    # numeric gradients via central finite differences
    eps = 1e-6
    numeric = np.zeros(len(vals))
    for i in range(len(vals)):
        up = list(vals)
        up[i] += eps
        dn = list(vals)
        dn[i] -= eps
        numeric[i] = (loss_from(up) - loss_from(dn)) / (2 * eps)

    max_err = np.max(np.abs(analytic - numeric))
    print("param   analytic      numeric       abs-err")
    for n, ga, gn in zip(names, analytic, numeric):
        print(f"  {n}   {ga:+.6f}   {gn:+.6f}   {abs(ga-gn):.2e}")
    print(f"\nmax abs error = {max_err:.2e}  ->  {'PASS' if max_err < 1e-4 else 'FAIL'}")

    # ---- figure: analytic vs numeric bars --------------------------------
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, analytic, w, label="analytic (backward)", color="#1d3557")
    ax.bar(x + w / 2, numeric, w, label="numeric (finite diff)", color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("gradient")
    ax.set_title(f"Gradient check: analytic vs numeric (max err {max_err:.1e})")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "grad_check.png", dpi=100)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'grad_check.png'}")


if __name__ == "__main__":
    main()
