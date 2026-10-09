"""Solution (a) — the forward process q_sample.

    uv run python solutions/ex_a_q_sample.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.schedules import Diffusion


def my_q_sample(diffusion: Diffusion, x0: torch.Tensor, t: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
    # gather per-example schedule constants and reshape to broadcast over x0.
    shape = (t.shape[0],) + (1,) * (x0.ndim - 1)
    sqrt_acp = diffusion.sqrt_alphas_cumprod.gather(0, t).reshape(shape)
    sqrt_om = diffusion.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(shape)
    return sqrt_acp * x0 + sqrt_om * noise


def main():
    torch.manual_seed(0)
    diffusion = Diffusion.from_schedule("cosine", timesteps=200)

    x0 = torch.randn(64, 2)
    t = torch.randint(0, 200, (64,))
    noise = torch.randn_like(x0)
    ref = diffusion.q_sample(x0, t, noise)
    got = my_q_sample(diffusion, x0, t, noise)
    err = (ref - got).abs().max().item()
    ok1 = err < 1e-6
    print(f"[test 1] matches reference q_sample: max abs err {err:.2e} -> "
          f"{'PASS' if ok1 else 'FAIL'}")

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
