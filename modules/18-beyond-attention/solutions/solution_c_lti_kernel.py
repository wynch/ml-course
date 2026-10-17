"""Exercise (c), solution: a fixed linear recurrence is a convolution.

LTIBlock's recurrence, per channel e and state n:
    h_t = a h_{t-1} + dt B x_t,   y_t = sum_n C h_t,   a = exp(dt A), A < 0
Unroll it: y_t = sum_{k>=0} K[k] x_{t-k} with K[k] = sum_n C B dt a^k.
Write the kernel, then the causal convolution, and check both against the
recurrence. Then make a selective version (a different a at every step) and
see that no single kernel fits.

Predict before you run: with dt = 0.05 and A = -1, the slowest of a channel's
decays, how much of an input is left in the state 100 steps later?

    uv run python ../solutions/solution_c_lti_kernel.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (sets the MPS guard before torch)
import torch


def kernel(a: torch.Tensor, B: torch.Tensor, C: torch.Tensor, dt: torch.Tensor, T: int) -> torch.Tensor:
    """a, B, C (E, N); dt (E,). Returns K (E, T), K[e, k] = sum_n C B dt a^k."""
    k = torch.arange(T, dtype=a.dtype)
    return ((C * B * dt[:, None])[..., None] * a[..., None] ** k).sum(1)


def causal_conv(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """x (B, T, E), K (E, T). y[b, t, e] = sum_{k <= t} K[e, k] x[b, t - k, e]."""
    T = x.shape[1]
    y = torch.zeros_like(x)
    for k in range(T):
        y[:, k:] += K[:, k] * x[:, : T - k]
    return y


def recurrence(x, a, B, C, dt):
    """The loop: h (B, E, N) updated once per step. a may be (E, N) or (B, T, E, N)."""
    h = torch.zeros(x.shape[0], *B.shape)
    ys = []
    for t in range(x.shape[1]):
        at = a if a.dim() == 2 else a[:, t]
        h = at * h + (dt * x[:, t])[..., None] * B
        ys.append((h * C).sum(-1))
    return torch.stack(ys, 1)


def main() -> None:
    torch.manual_seed(0)
    E, N, T = 8, 16, 64
    A = -torch.arange(1, N + 1).float().repeat(E, 1)
    dt = torch.full((E,), 0.05)
    a = torch.exp(dt[:, None] * A)
    B, C = torch.randn(E, N), torch.randn(E, N)
    x = torch.randn(2, T, E)

    K = kernel(a, B, C, dt, T)
    print(f"max |convolution - recurrence|: {(causal_conv(x, K) - recurrence(x, a, B, C, dt)).abs().max():.2e}")
    print(f"slowest decay a = exp(0.05 * -1) = {math.exp(-0.05):.4f}; after 100 steps {math.exp(-0.05) ** 100:.4f} is left")

    a_sel = a * torch.rand(2, T, E, N).clamp(min=0.5)       # a different decay at every step
    y_sel = recurrence(x, a_sel, B, C, dt)
    print(f"selective: max |convolution with the fixed kernel - recurrence| = "
          f"{(causal_conv(x, K) - y_sel).abs().max():.2f}: no single kernel")


if __name__ == "__main__":
    main()
