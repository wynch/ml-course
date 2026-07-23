"""Exercise (c) — conditional generation in 2D with classifier-free guidance.

We train ONE model on the two-moons data that can be told which moon to draw.
The trick (classifier-free guidance, CFG) is to train the model both *with* the
class label and, a fraction of the time, with a "null" label — then at sampling
time combine the two predictions to steer generation:

    eps = eps_uncond + w * (eps_cond - eps_uncond)

w=0 ignores the label, w=1 is plain conditional, w>1 amplifies the class.
Implement `guided_eps` below; this script trains the conditional model and
renders generation of each moon across several guidance scales.

    uv run python exercises/ex_c_conditional.py

Output: figures/ex_c_conditional.png
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.embeddings import get_device
from src.schedules import Diffusion
from src.toy2d import DenoiseMLP, make_moons, train_toy

FIG = pathlib.Path(__file__).resolve().parents[1].parent / "figures"


def guided_eps(eps_uncond: torch.Tensor, eps_cond: torch.Tensor,
               w: float) -> torch.Tensor:
    """Combine unconditional and conditional noise predictions with CFG scale w."""
    # TODO(you): return eps_uncond + w * (eps_cond - eps_uncond)
    raise NotImplementedError("implement guided_eps")


@torch.no_grad()
def sample(model, diffusion, n, device, cls, w):
    x = torch.randn(n, 2, device=device)
    y_c = torch.full((n,), cls, device=device, dtype=torch.long)
    y_n = torch.full((n,), model.num_classes, device=device, dtype=torch.long)
    for t in reversed(range(diffusion.timesteps)):
        tb = torch.full((n,), t, device=device, dtype=torch.long)
        eps = guided_eps(model(x, tb, y=y_n), model(x, tb, y=y_c), w)
        beta = diffusion.betas[t]
        mean = (x - beta / diffusion.sqrt_one_minus_alphas_cumprod[t] * eps) \
            / torch.sqrt(diffusion.alphas[t])
        x = mean if t == 0 else mean + torch.sqrt(beta) * torch.randn_like(x)
    return x.cpu().numpy()


def main():
    device = get_device()
    data, labels = make_moons(2000, seed=0)
    diffusion = Diffusion.from_schedule("cosine", 200, device=device)
    model = DenoiseMLP(num_classes=2)
    print("training conditional model (10% label dropout for CFG)...")
    train_toy(model, diffusion, data, labels=labels, cfg_drop=0.1,
              epochs=3000, device=device, log_every=500)

    scales = [0.0, 1.0, 3.0]
    fig, axes = plt.subplots(len(scales), 2, figsize=(6, 3 * len(scales)))
    for r, w in enumerate(scales):
        for cls in (0, 1):
            ax = axes[r, cls]
            ax.scatter(data[:, 0], data[:, 1], s=4, c="#cccccc", alpha=0.5)
            pts = sample(model, diffusion, 800, device, cls, w)
            ax.scatter(pts[:, 0], pts[:, 1], s=5,
                       c="#3b6ea5" if cls == 0 else "#b5452f", alpha=0.6)
            ax.set_title(f"class {cls}, guidance w={w}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Conditional 2D generation with classifier-free guidance", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "ex_c_conditional.png", dpi=90); plt.close(fig)
    print("wrote ex_c_conditional.png")


if __name__ == "__main__":
    main()
