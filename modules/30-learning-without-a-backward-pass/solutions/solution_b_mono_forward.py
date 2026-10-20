"""Solution (b): one Mono-Forward layer, its local loss and gradients, by hand.

    uv run python ../solutions/solution_b_mono_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import common  # noqa: F401  (laptop guard before torch)
import torch


def layer_grads(w, b, m, x, y):
    """a = relu(x Wᵀ + b); local logits a Mᵀ; mean cross-entropy against y.
    Returns (loss, dW, db, dM, a). Nothing flows back into x."""
    z = x @ w.T + b
    a = torch.relu(z)
    logits = a @ m.T
    p = torch.softmax(logits, dim=1)
    loss = float(-torch.log(p[torch.arange(len(y)), y]).mean())
    dlog = p.clone()
    dlog[torch.arange(len(y)), y] -= 1.0
    dlog /= len(y)
    dm = dlog.T @ a
    dz = (dlog @ m) * (z > 0)
    return loss, dz.T @ x, dz.sum(0), dm, a


def main() -> None:
    import rules as R

    x_tr, y_tr, x_te, y_te = common.load_fashion()
    mf = R.MonoForward(R.SIZES, 0)
    g = torch.Generator().manual_seed(0)
    for _ in range(2):
        perm = torch.randperm(len(x_tr), generator=g)
        for i in range(0, len(x_tr) - 63, 64):
            idx = perm[i:i + 64]
            a = x_tr[idx]
            for (w, b), m, opt in zip(mf.layers, mf.heads, mf.opts):
                _, gw, gb, gm, a = layer_grads(w, b, m, a, y_tr[idx])
                opt.step([gw, gb, gm])
    for k in range(3):
        acc = float((mf.predict(x_te, layer=k).argmax(1) == y_te).float().mean())
        print(f"after 2 epochs, the head of hidden layer {k + 1} scores {acc:.3f} on the test set")


if __name__ == "__main__":
    main()
