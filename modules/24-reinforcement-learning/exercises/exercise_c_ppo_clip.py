"""Exercise (c): PPO's clipped objective and its clip fraction.

For each sample: the ratio rho = exp(logp_new - logp_old) between the new
policy's probability of the action taken and the old one's, and an advantage
A (how much better than expected the action turned out). PPO maximises

    mean( min( rho A, clip(rho, 1 - eps, 1 + eps) A ) )

The clip fraction is the share of samples with |rho - 1| > eps.

Predict before you run: take one sample with A = +1. For which ratios does
the objective still give a gradient that pushes rho up? And with A = -1, for
which ratios does it still push rho down? Fill in the ``# TODO(you):``
lines, then:

    uv run python ../exercises/exercise_c_ppo_clip.py
    uv run pytest tests/test_solutions.py -k clip -v
"""

from __future__ import annotations

import torch


def clipped_objective(logp_new, logp_old, adv, clip=0.2):
    """Returns (objective to maximise, clip fraction, ratio)."""
    # TODO(you): ratio = exp(logp_new - logp_old)
    # TODO(you): obj = mean of min(ratio * adv, clamp(ratio, 1 - clip, 1 + clip) * adv)
    # TODO(you): clip_frac = share of samples with |ratio - 1| > clip, as a float tensor
    ...


def main() -> None:
    for adv in (1.0, -1.0):
        for rho in (0.7, 0.9, 1.0, 1.1, 1.3):
            logp_new = torch.tensor([float(torch.log(torch.tensor(rho)))], requires_grad=True)
            obj, frac, _ = clipped_objective(logp_new, torch.zeros(1), torch.tensor([adv]))
            obj.backward()
            print(f"A {adv:+.0f}, ratio {rho:.1f}: objective {obj.item():+.3f}, "
                  f"gradient on log pi_new {logp_new.grad.item():+.3f}, clipped {frac.item():.0f}")


if __name__ == "__main__":
    main()
