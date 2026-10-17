"""Section 4: linear CKA between networks.

Two copies of module 02's MLP, seeds 0 and 1, same data and recipe. Do they hold the
same geometry? Linear CKA compares two sets of activations on the same inputs, up to
rotation and overall scale. A per-neuron match is shown beside it.

    uv run python src/cka.py --figures
"""

from __future__ import annotations

import argparse

import common
import numpy as np
import torch

N_INPUTS = 5_000                  # test images the activations are taken on


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between activations x (n, p) and y (n, q) on the same n inputs.

    CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), columns centred.
    1 when y is a rotation and rescaling of x; near 0 when they share no linear structure.
    """
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    num = np.linalg.norm(yc.T @ xc) ** 2
    return float(num / (np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)))


def best_neuron_match(x: np.ndarray, y: np.ndarray) -> float:
    """For each unit of x, its best |correlation| with any unit of y; the median over units of x."""
    def z(a):
        a = a - a.mean(0)
        s = a.std(0)
        return a[:, s > 0] / s[s > 0]
    c = np.abs(z(x).T @ z(y)) / len(x)
    return float(np.median(c.max(1)))


def run() -> dict:
    x_tr, y_tr = common.load_fashion("train")
    x_te, y_te = common.load_fashion("test")
    x = x_te[:N_INPUTS]
    nets = {}
    for seed in (0, 1):
        model, init = common.train_m02_mlp(x_tr, y_tr, seed=seed)
        acc = float(common._m02().nn.accuracy(model, x_te, y_te))
        nets[f"trained {seed}"] = common.mlp_layers(common.mlp_weights(model), x)
        nets[f"init {seed}"] = common.mlp_layers(init, x)
        nets[f"acc {seed}"] = round(acc, 4)
        print(f"seed {seed}: test accuracy {acc:.4f}")
    px = x.astype(np.float64)
    pairs = [("trained 0", "trained 1"), ("init 0", "init 1"), ("trained 0", "init 0"), ("trained 0", "init 1")]
    rows = []
    for a, b in pairs:
        row = {"a": a, "b": b}
        for li, layer in enumerate(("hidden", "logits")):
            row[f"cka_{layer}"] = round(linear_cka(nets[a][li], nets[b][li]), 4)
        row["neuron_match_hidden"] = round(best_neuron_match(nets[a][0], nets[b][0]), 4)
        rows.append(row)
        print(f"{a} vs {b}: CKA hidden {row['cka_hidden']}, logits {row['cka_logits']}, "
              f"best neuron match {row['neuron_match_hidden']}")
    to_pixels = {k: round(linear_cka(px, nets[k][0]), 4) for k in ("trained 0", "init 0")}
    print("CKA pixels vs hidden:", to_pixels)
    rng = np.random.default_rng(0)
    h = nets["trained 0"][0]
    q, _ = np.linalg.qr(rng.normal(size=(256, 256)))
    checks = {"rotated_and_scaled": round(linear_cka(h, 3.0 * h @ q), 6),
              "permuted": round(linear_cka(h, h[:, rng.permutation(256)]), 6),
              "noise_vs_hidden": round(linear_cka(h, rng.normal(size=h.shape)), 4)}
    print("sanity:", checks)
    return {"n_inputs": N_INPUTS, "test_acc": {"seed 0": nets["acc 0"], "seed 1": nets["acc 1"]},
            "pairs": rows, "pixels_vs_hidden": to_pixels, "checks": checks}


def fig_cka(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = r["pairs"]
    labels = [f"{p['a']}\nvs {p['b']}" for p in rows]
    xs = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(xs - 0.27, [p["cka_hidden"] for p in rows], 0.27, color="#1d6fb8", label="CKA, hidden layer")
    ax.bar(xs, [p["cka_logits"] for p in rows], 0.27, color="#5aa1e6", label="CKA, logits")
    ax.bar(xs + 0.27, [p["neuron_match_hidden"] for p in rows], 0.27, color="0.7", label="best single-neuron match")
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Module 02's MLP, two seeds, {r['n_inputs']:,} test images", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("cka.py")
    torch.set_num_threads(4)
    res = run()
    res["guard"] = guard.report()
    common.write_results("cka", res)
    if args.figures:
        fig_cka(res, common.FIGURES / "cka.png")


if __name__ == "__main__":
    main()
