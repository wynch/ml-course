"""The XOR wall, and the smallest thing that climbs it.

Three pictures in one figure: every line fails on XOR (exhaustive search over
directions), the perceptron cycles forever instead of converging, and a 2→2→1
tanh network solves it — because its hidden layer bends the plane first.

Produces:
  figures/xor_wall.png
  run_xor.json

Run:  uv run scripts/xor_wall.py
"""

import json
import pathlib

import _bootstrap  # noqa: F401  (sets sys.path + FIGDIR)
from _bootstrap import FIGDIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perceptron import mlp, rosenblatt
from perceptron.data import augment, xor_data

OUT = pathlib.Path(__file__).resolve().parents[1] / "run_xor.json"
POS = "#1f918d"
NEG = "#453781"
HIDDEN = 2
SEED = 7
LR = 0.5
STEPS = 4000


def best_line_accuracy(X, y, n_dirs=720, n_offsets=400):
    """Exhaustive search: the best any single linear threshold can do on XOR.

    Ties on accuracy are broken by the margin, so the line we draw is the most
    confident of the many that reach the ceiling.
    """
    best = (0.0, -np.inf)
    best_wb = None
    for theta in np.linspace(0, np.pi, n_dirs, endpoint=False):
        w = np.array([np.cos(theta), np.sin(theta)])
        proj = X @ w
        for b in np.linspace(proj.min() - 1.0, proj.max() + 1.0, n_offsets):
            for sgn in (1.0, -1.0):
                margins = sgn * (proj - b) * y
                acc = float((margins > 0).mean())
                key = (acc, float(np.min(np.abs(proj - b))))
                if key > best:
                    best, best_wb = key, (sgn * w, -sgn * b)
    return best[0], best_wb


def main() -> None:
    X, y = xor_data()
    Xa = augment(X)

    best_acc, (wb_w, wb_b) = best_line_accuracy(X, y)
    print(f"XOR       points {X.tolist()}  labels {y.tolist()}")
    print(f"best line {best_acc*100:.0f}% of 4 points "
          f"(exhaustive over 720 directions × 400 offsets)")
    assert best_acc == 0.75, best_acc

    run = rosenblatt.train(Xa, y, max_epochs=500)
    print(f"perceptron converged={run.converged}  mistakes={run.mistakes} "
          f"in {run.epochs} epochs  accuracy {run.accuracy(Xa, y)*100:.0f}%")
    print(f"          mistakes per epoch, first 8: {run.per_epoch[:8]} — "
          "a cycle, not a convergence")
    assert not run.converged

    net = mlp.TinyMLP.init(2, HIDDEN, seed=SEED)
    net, losses, accs = mlp.train(net, X, y.astype(float), lr=LR, steps=STEPS)
    H, out = net.forward(X)
    pred = net.predict(X)
    first_perfect = int(np.argmax(np.array(accs) == 1.0))
    print(f"tiny MLP  2→{HIDDEN}→1 tanh, {net.n_params()} parameters, "
          f"lr={LR}, {STEPS} full-batch steps")
    print(f"          loss {losses[0]:.4f} → {losses[-1]:.3e}   "
          f"accuracy {accs[0]*100:.0f}% → {accs[-1]*100:.0f}% "
          f"(first perfect at step {first_perfect})")
    print(f"          outputs {np.round(out, 6).tolist()}  vs targets {y.tolist()}")
    print(f"          w cycle on XOR: "
          f"{[s.tolist() for s in run.snapshots[:5]]} — period 4, because Σ yᵢxᵢ = 0")
    assert (pred == y).all()

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))

    ax = axes[0]
    xs = np.linspace(-0.55, 1.55, 40)
    rng = np.random.default_rng(3)
    for theta in np.linspace(0, np.pi, 26, endpoint=False):
        w = np.array([np.cos(theta), np.sin(theta)])
        b = rng.uniform(-0.4, 1.4)
        if abs(w[1]) < 1e-6:
            ax.axvline(b / w[0], color="#c1121f", lw=0.6, alpha=0.28)
        else:
            ax.plot(xs, (b - w[0] * xs) / w[1], color="#c1121f", lw=0.6, alpha=0.28)
    ax.plot(xs, -(wb_w[0] * xs + wb_b) / wb_w[1], color="#c1121f", lw=2.2,
            label=f"best possible line: {best_acc*100:.0f}%")
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=180, c=POS, edgecolors="white",
               zorder=4, label="+1")
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=180, c=NEG, edgecolors="white",
               zorder=4, label="−1")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_aspect("equal")
    ax.set_title("no line separates XOR\n(the two +1s are diagonal neighbours)",
                 fontsize=10)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    snaps = np.array(run.snapshots[:13])
    for j, (lab, col) in enumerate((("w₁", "#1f918d"), ("w₂", "#453781"),
                                    ("b", "#a86b0e"))):
        ax.plot(range(len(snaps)), snaps[:, j], "-o", color=col, lw=1.8, ms=4,
                label=lab)
    for k in range(0, len(snaps), 4):
        ax.axvline(k, color="#5c6b67", lw=0.7, ls=":", alpha=0.8)
    ax.set_xlabel("update number")
    ax.set_ylabel("weight value")
    ax.set_title("so the perceptron never stops: w returns to 0\n"
                 f"every 4 updates — {run.mistakes} of them in {run.epochs} epochs",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=3, loc="lower center")

    ax = axes[2]
    gx, gy = np.meshgrid(np.linspace(-0.55, 1.55, 260),
                         np.linspace(-0.55, 1.55, 260))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    Z = net.forward(G)[1].reshape(gx.shape)
    cs = ax.contourf(gx, gy, Z, levels=np.linspace(-1.35, 1.35, 28), cmap="viridis")
    ax.contour(gx, gy, Z, levels=[0.0], colors="white", linewidths=2.0)
    fig.colorbar(cs, ax=ax, label="network output")
    ax.scatter(X[:, 0], X[:, 1], s=200, c="white", edgecolors="#1c2422",
               linewidths=1.6, zorder=4)
    for (px, py), lab in zip(X, y):
        ax.text(px, py, "+1" if lab > 0 else "−1", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1c2422", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"one hidden layer ({net.n_params()} parameters)\n"
                 f"loss {losses[-1]:.1e}, 4/4 correct", fontsize=10)

    fig.suptitle("The XOR wall (1969) and the hidden layer that goes over it",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "xor_wall.png", dpi=110)
    plt.close(fig)

    blob = {
        "best_line_accuracy": best_acc,
        "perceptron": {"converged": run.converged, "mistakes": run.mistakes,
                       "epochs": run.epochs, "per_epoch_first8": run.per_epoch[:8],
                       "accuracy": run.accuracy(Xa, y)},
        "mlp": {
            "hidden": HIDDEN, "seed": SEED, "lr": LR, "steps": STEPS,
            "n_params": net.n_params(),
            "loss_start": float(losses[0]), "loss_final": float(losses[-1]),
            "acc_final": float(accs[-1]), "first_perfect_step": first_perfect,
            "outputs": [float(v) for v in out],
            "W1": net.W1.tolist(), "b1": net.b1.tolist(),
            "W2": net.W2.tolist(), "b2": net.b2.tolist(),
            "hidden_activations": H.tolist(),
        },
    }
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"wrote {FIGDIR / 'xor_wall.png'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
