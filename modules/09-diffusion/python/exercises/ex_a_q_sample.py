"""Exercise (a) — implement the forward process q_sample.

The forward (noising) process has a closed form: you can jump straight to any
timestep t without simulating the steps in between.

    x_t = sqrt(alpha_bar_t) * x0  +  sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

where alpha_bar_t (`alphas_cumprod[t]`) is the cumulative product of (1 - beta).

Fill in `my_q_sample` below. Run this file: it checks your output against the
reference implementation and against the theoretical marginal statistics.

    uv run python exercises/ex_a_q_sample.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.schedules import Diffusion


def my_q_sample(diffusion: Diffusion, x0: torch.Tensor, t: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
    """Return x_t given clean data x0, timesteps t (shape (B,)), and noise.

    Useful precomputed tensors on `diffusion` (each shape (timesteps,)):
      diffusion.sqrt_alphas_cumprod
      diffusion.sqrt_one_minus_alphas_cumprod
    Index them at `t` and broadcast over the data dimensions of x0.
    """
    # TODO(you): implement the closed-form forward process.
    #   1. gather sqrt_alphas_cumprod[t] and sqrt_one_minus_alphas_cumprod[t]
    #   2. reshape each to (B, 1, 1, ...) so it broadcasts against x0
    #   3. return sqrt_acp * x0 + sqrt_om * noise
    raise NotImplementedError("implement my_q_sample")


def main():
    torch.manual_seed(0)
    diffusion = Diffusion.from_schedule("cosine", timesteps=200)

    # Test 1: matches the reference implementation exactly (same noise).
    x0 = torch.randn(64, 2)
    t = torch.randint(0, 200, (64,))
    noise = torch.randn_like(x0)
    ref = diffusion.q_sample(x0, t, noise)
    got = my_q_sample(diffusion, x0, t, noise)
    err = (ref - got).abs().max().item()
    ok1 = err < 1e-6
    print(f"[test 1] matches reference q_sample: max abs err {err:.2e} -> "
          f"{'PASS' if ok1 else 'FAIL'}")

    # Test 2: at t=0 output is ~x0; at t=T-1 it is ~unit-variance noise.
    big = torch.randn(4000, 2)
    x_last = my_q_sample(diffusion, big, torch.full((4000,), 199), torch.randn_like(big))
    var = x_last.var().item()
    ok2 = 0.7 < var < 1.3
    print(f"[test 2] variance at t=199 is ~1: {var:.3f} -> "
          f"{'PASS' if ok2 else 'FAIL'}")

    print("ALL PASS" if (ok1 and ok2) else "SOME TESTS FAILED")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
