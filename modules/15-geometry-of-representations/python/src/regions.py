"""Section 1: the linear regions of a ReLU network.

A ReLU net is linear inside each region where no neuron changes sign. Two parts:

1. A 2 -> 12 -> 12 -> 2 net on two moons. We cut the square [-2.5, 2.5]^2 into its
   regions exactly (convex polygons, no sampling), at init and after training.
2. 784 -> w -> w -> 10 nets on FashionMNIST, w = 32, 64, 128, 256, one seed: the
   number of regions along 30 straight lines between two test images, counted
   exactly, before and after 5 epochs of training. A demo, not a measurement.

    uv run python src/regions.py --figures
"""

from __future__ import annotations

import argparse

import common
import numpy as np
import torch

BOX = 2.5
TOY_WIDTH = 12
TOY_SEED = 0
WIDTHS = (32, 64, 128, 256)
N_LINES = 30
EPOCHS = 5


# ------------------------------------------------------------------ exact 2-D tiling

def split(poly: np.ndarray, a: np.ndarray, c: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cut a convex polygon (k, 2) by the line a.x + c = 0. Returns (side > 0, side < 0)."""
    s = poly @ a + c
    if np.all(s >= 0):
        return poly, None
    if np.all(s <= 0):
        return None, poly
    pos, neg = [], []
    k = len(poly)
    for i in range(k):
        p, q, sp, sq = poly[i], poly[(i + 1) % k], s[i], s[(i + 1) % k]
        if sp >= 0:
            pos.append(p)
        if sp <= 0:
            neg.append(p)
        if (sp > 0 > sq) or (sp < 0 < sq):
            x = p + (q - p) * (sp / (sp - sq))
            pos.append(x)
            neg.append(x)
    return np.array(pos), np.array(neg)


def area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def tile(weights, box: float = BOX, min_area: float = 1e-10) -> list[np.ndarray]:
    """Exact linear regions of a 2 -> w -> w -> . ReLU net inside the square [-box, box]^2.

    Layer 1 cuts the square by w straight lines. Inside each of those pieces layer 1 is a
    fixed linear map, so each layer-2 neuron is linear too and cuts the piece by one more line.
    """
    (w1, b1), (w2, b2) = weights[0], weights[1]          # w1 (2, w), w2 (w, w)
    square = np.array([[-box, -box], [box, -box], [box, box], [-box, box]], dtype=np.float64)
    cells = [square]
    for j in range(w1.shape[1]):
        nxt = []
        for p in cells:
            for half in split(p, w1[:, j], b1[j]):
                if half is not None and len(half) >= 3 and area(half) > min_area:
                    nxt.append(half)
        cells = nxt
    out = []
    for p in cells:
        mask = (p.mean(0) @ w1 + b1) > 0                  # layer-1 pattern of this piece
        a2 = (w1 * mask) @ w2                              # pre-activation 2 = x @ a2 + c2 inside it
        c2 = (b1 * mask) @ w2 + b2
        sub = [p]
        for j in range(w2.shape[1]):
            nxt = []
            for q in sub:
                for half in split(q, a2[:, j], c2[j]):
                    if half is not None and len(half) >= 3 and area(half) > min_area:
                        nxt.append(half)
            sub = nxt
        out.extend(sub)
    return out


def moons(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved half circles, standardised (the scikit-learn recipe, written out)."""
    rng = np.random.default_rng(seed)
    h = n // 2
    t1, t2 = rng.uniform(0, np.pi, h), rng.uniform(0, np.pi, n - h)
    x = np.concatenate([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    x += rng.normal(0, noise, x.shape)
    y = np.r_[np.zeros(h), np.ones(n - h)].astype(np.int64)
    return ((x - x.mean(0)) / x.std(0)).astype(np.float32), y


def toy_net(width: int, seed: int) -> torch.nn.Sequential:
    torch.manual_seed(seed)                                # PyTorch's default init: biases are not zero
    return torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 2))


def net_weights(net) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(W (in, out), b)] per Linear, float64, so that a layer is x @ W + b."""
    return [(m.weight.detach().double().numpy().T.copy(), m.bias.detach().double().numpy().copy())
            for m in net if isinstance(m, torch.nn.Linear)]


def run_toy() -> dict:
    x, y = moons()
    net = toy_net(TOY_WIDTH, TOY_SEED)
    w_init = net_weights(net)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(2000):
        loss = torch.nn.functional.cross_entropy(net(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = float((net(xt).argmax(1) == yt).float().mean())
    w_tr = net_weights(net)
    cells_init, cells_tr = tile(w_init), tile(w_tr)
    print(f"toy 2-{TOY_WIDTH}-{TOY_WIDTH}-2: {len(cells_init)} regions at init, {len(cells_tr)} trained "
          f"(train acc {acc:.3f}, loss {loss.item():.3f})")
    rnd = lambda w: [[np.round(W, 5).tolist(), np.round(b, 5).tolist()] for W, b in w]
    return {"width": TOY_WIDTH, "seed": TOY_SEED, "box": BOX, "train_acc": round(acc, 4),
            "final_loss": round(loss.item(), 4), "regions_init": len(cells_init), "regions_trained": len(cells_tr),
            "weights_init": rnd(w_init), "weights_trained": rnd(w_tr),
            "points": np.round(x, 3).tolist(), "labels": y.tolist()}, (cells_init, cells_tr)


# ------------------------------------------------------------------ exact count along a line

def count_on_segment(weights, a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Exact regions met by the segment a -> b in a ReLU net with two hidden layers.

    Returns (pieces cut by layer 1 alone, pieces cut by both layers). Along the segment every
    pre-activation of layer 1 is linear in t; inside each layer-1 piece, so is layer 2's.
    """
    (w1, b1), (w2, b2) = weights[0], weights[1]
    ha, hb = a @ w1 + b1, b @ w1 + b1
    cross = (ha > 0) != (hb > 0)
    t1 = ha[cross] / (ha[cross] - hb[cross])
    t1 = np.unique(t1[(t1 > 0) & (t1 < 1)])
    edges = np.concatenate([[0.0], t1, [1.0]])
    t2 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (((1 - (lo + hi) / 2) * ha + (lo + hi) / 2 * hb) > 0)
        pre = lambda t: (((1 - t) * ha + t * hb) * mask) @ w2 + b2
        p, q = pre(lo), pre(hi)
        c = (p > 0) != (q > 0)
        t = lo + (hi - lo) * p[c] / (p[c] - q[c])
        t2.extend(t[(t > lo) & (t < hi)])
    return len(t1) + 1, len(np.unique(np.concatenate([t1, np.array(t2)]))) + 1


def fashion_net(width: int, seed: int = 0) -> torch.nn.Sequential:
    """784 -> w -> w -> 10, initialised like module 02's Linear: N(0, 2 / fan_in), zero bias."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(784, width), torch.nn.ReLU(),
                              torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 10))
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, (2.0 / m.in_features) ** 0.5)
            torch.nn.init.zeros_(m.bias)
    return net


def train_fashion(net, x, y, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            loss = torch.nn.functional.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def run_width(guard) -> dict:
    x_tr, y_tr = common.load_fashion("train")
    x_te, y_te = common.load_fashion("test")
    guard.tick("load")
    rng = np.random.default_rng(1)
    pairs = [tuple(int(v) for v in rng.choice(len(x_te), 2, replace=False)) for _ in range(N_LINES)]
    xd = x_te.astype(np.float64)
    out = []
    for w in WIDTHS:
        net = fashion_net(w)
        init = [count_on_segment(net_weights(net), xd[i], xd[j]) for i, j in pairs]
        train_fashion(net, x_tr, y_tr)
        with torch.no_grad():
            acc = float((net(torch.from_numpy(x_te)).argmax(1).numpy() == y_te).mean())
        tr = [count_on_segment(net_weights(net), xd[i], xd[j]) for i, j in pairs]
        row = {"width": w, "neurons": 2 * w, "test_acc": round(acc, 4),
               "init_mean": round(float(np.mean([c for _, c in init])), 2),
               "trained_mean": round(float(np.mean([c for _, c in tr])), 2),
               "init_layer1_mean": round(float(np.mean([c for c, _ in init])), 2),
               "trained_layer1_mean": round(float(np.mean([c for c, _ in tr])), 2),
               "trained_counts": [c for _, c in tr], "init_counts": [c for _, c in init]}
        out.append(row)
        guard.tick(f"width {w}")
        print(f"w={w}: acc {acc:.3f}, regions per line {row['init_mean']} at init, {row['trained_mean']} trained")
    ratios = [round(b["trained_mean"] / a["trained_mean"], 3) for a, b in zip(out, out[1:])]
    return {"widths": out, "pairs": pairs, "ratio_per_doubling_trained": ratios,
            "ratio_per_doubling_init": [round(b["init_mean"] / a["init_mean"], 3) for a, b in zip(out, out[1:])],
            "lines": N_LINES, "epochs": EPOCHS, "seed": 0}


# ------------------------------------------------------------------ figure

def fig_regions(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    toy = r["toy"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    rng = np.random.default_rng(0)
    pts, lab = np.array(toy["points"]), np.array(toy["labels"])
    for ax, key, title in ((axes[0], "weights_init", "at init"), (axes[1], "weights_trained", "after training")):
        w = [(np.array(W), np.array(b)) for W, b in toy[key]]
        cells = tile(w, toy["box"])
        cols = plt.cm.Pastel1(rng.integers(0, 9, len(cells)))
        ax.add_collection(PolyCollection(cells, facecolors=cols, edgecolors="0.35", linewidths=0.4))
        ax.scatter(pts[:, 0], pts[:, 1], c=np.where(lab == 0, "#1d6fb8", "#b7410e"), s=5, zorder=3)
        ax.set_xlim(-toy["box"], toy["box"])
        ax.set_ylim(-toy["box"], toy["box"])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"2-{toy['width']}-{toy['width']}-2 net {title}: {len(cells)} regions", fontsize=10)
    ax = axes[2]
    ws = r["width"]["widths"]
    n = [row["neurons"] for row in ws]
    ax.plot(n, [row["trained_mean"] for row in ws], "o-", color="#1d6fb8", label="trained, 5 epochs")
    ax.plot(n, [row["init_mean"] for row in ws], "s--", color="#888888", label="at init")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("hidden neurons (2 × width)")
    ax.set_ylabel(f"regions along a line (mean of {r['width']['lines']})")
    ax.set_title("784-w-w-10 on FashionMNIST: lines between test images", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("regions.py")
    torch.set_num_threads(4)
    toy, _ = run_toy()
    guard.tick("toy")
    width = run_width(guard)
    res = {"toy": toy, "width": width, "guard": guard.report()}
    common.write_results("regions", res)
    if args.figures:
        fig_regions(res, common.FIGURES / "regions.png")


if __name__ == "__main__":
    main()
