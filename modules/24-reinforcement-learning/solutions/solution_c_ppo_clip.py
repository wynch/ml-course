"""Reference solution, exercise (c): PPO's clipped objective and its clip fraction."""

from __future__ import annotations

import torch


def clipped_objective(logp_new, logp_old, adv, clip=0.2):
    """Returns (objective to maximise, clip fraction, ratio)."""
    ratio = torch.exp(logp_new - logp_old)
    obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
    clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return obj, clip_frac, ratio


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
