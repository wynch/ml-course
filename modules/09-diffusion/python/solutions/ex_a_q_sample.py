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

    # Test 2: data far from the origin (every point at (5, 5)). At t=0 it stays
    # put; at t=T-1 the mean is gone and the variance is ~1. Unit-variance x0
    # would not do here: then x_t has variance 1 at every t, so returning x0
    # or the noise would pass too.
    far = torch.full((4000, 2), 5.0)
    x_first = my_q_sample(diffusion, far, torch.zeros(4000, dtype=torch.long),
                          torch.randn_like(far))
    x_last = my_q_sample(diffusion, far, torch.full((4000,), 199),
                         torch.randn_like(far))
    m0, mT, vT = x_first.mean().item(), x_last.mean().item(), x_last.var().item()
    ok2 = abs(m0 - 5.0) < 0.1 and abs(mT) < 0.1 and 0.9 < vT < 1.1
    print(f"[test 2] mean at t=0 {m0:.3f} (want ~5); at t=199 mean {mT:.3f} "
          f"(want ~0), variance {vT:.3f} (want ~1) -> {'PASS' if ok2 else 'FAIL'}")

    print("ALL PASS" if (ok1 and ok2) else "SOME TESTS FAILED")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
