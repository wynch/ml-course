"""Section 1a: Bradley-Terry by hand on synthetic pairs.

Twelve items get a hidden score s. A simulated judge compares random pairs
and prefers i over j with probability sigmoid(s_i - s_j). From the outcomes
alone, gradient descent on the Bradley-Terry loss recovers the scores (up to
a shift: only differences matter, so we fix the mean at zero).

    uv run python src/bt_synthetic.py --figures

Runs on the CPU in about a second; numpy only, gradient written out.
"""

from __future__ import annotations

import argparse

from common import FIGURES, Guard, write_results

import numpy as np

N_ITEMS = 12
SIZES = (30, 100, 300, 1000, 3000, 10000)
SEEDS = 20
MAIN_SIZE = 1000
MAIN_SEED = 0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def true_scores() -> np.ndarray:
    rng = np.random.default_rng(1234)
    s = np.sort(rng.normal(0, 1.2, N_ITEMS))[::-1]
    return s - s.mean()


def simulate(s: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n comparisons of random distinct pairs: returns (winner, loser) index arrays."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), n)
    j = (i + rng.integers(1, len(s), n)) % len(s)          # j != i
    i_wins = rng.random(n) < sigmoid(s[i] - s[j])
    return np.where(i_wins, i, j), np.where(i_wins, j, i)


def bt_loss(s: np.ndarray, w: np.ndarray, l: np.ndarray) -> float:
    """Mean negative log-likelihood: -log sigmoid(s_w - s_l)."""
    return float(np.mean(np.logaddexp(0.0, -(s[w] - s[l]))))


def bt_grad(s: np.ndarray, w: np.ndarray, l: np.ndarray) -> np.ndarray:
    """d loss / d s. For one pair, d/ds_w = -(1 - sigmoid(s_w - s_l)) and d/ds_l is its negative."""
    g_pair = -(1.0 - sigmoid(s[w] - s[l])) / len(w)
    g = np.zeros_like(s)
    np.add.at(g, w, g_pair)
    np.add.at(g, l, -g_pair)
    return g


def fit_bt(w: np.ndarray, l: np.ndarray, n_items: int, steps: int = 4000, lr: float = 4.0,
           l2: float = 1e-4) -> tuple[np.ndarray, list[float]]:
    """Plain gradient descent from zero. A small L2 keeps an item that never lost finite."""
    s = np.zeros(n_items)
    curve = []
    for t in range(steps):
        s -= lr * (bt_grad(s, w, l) + l2 * s)
        s -= s.mean()                                      # only differences are identified
        if t % 50 == 0 or t == steps - 1:
            curve.append(bt_loss(s, w, l))
    return s, curve


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            num += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(num / (n * (n - 1) / 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("bt_synthetic")
    s_true = true_scores()

    w, l = simulate(s_true, MAIN_SIZE, MAIN_SEED)
    s_fit, curve = fit_bt(w, l, N_ITEMS)
    main_run = {
        "comparisons": MAIN_SIZE, "seed": MAIN_SEED,
        "s_fit": [round(float(x), 4) for x in s_fit],
        "kendall_tau": round(kendall_tau(s_true, s_fit), 4),
        "max_abs_error": round(float(np.abs(s_fit - s_true).max()), 4),
        "rmse": round(float(np.sqrt(np.mean((s_fit - s_true) ** 2))), 4),
        "loss_at_true": round(bt_loss(s_true, w, l), 4),
        "loss_at_fit": round(bt_loss(s_fit, w, l), 4),
        "loss_at_zero": round(bt_loss(np.zeros(N_ITEMS), w, l), 4),
        "loss_curve_every_50": [round(c, 4) for c in curve],
        "rank_exact": bool(np.array_equal(np.argsort(-s_fit), np.argsort(-s_true))),
    }
    print(f"{MAIN_SIZE} comparisons: Kendall tau {main_run['kendall_tau']}, RMSE {main_run['rmse']}, "
          f"loss fit {main_run['loss_at_fit']} vs true {main_run['loss_at_true']} vs zero {main_run['loss_at_zero']}")

    sweep = []
    for n in SIZES:
        taus, rmses, exact = [], [], 0
        for seed in range(SEEDS):
            ww, ll = simulate(s_true, n, 100 + seed)
            sf, _ = fit_bt(ww, ll, N_ITEMS)
            taus.append(kendall_tau(s_true, sf))
            rmses.append(float(np.sqrt(np.mean((sf - s_true) ** 2))))
            exact += int(np.array_equal(np.argsort(-sf), np.argsort(-s_true)))
        sweep.append({"comparisons": n, "kendall_tau_mean": round(float(np.mean(taus)), 4),
                      "kendall_tau_min": round(float(np.min(taus)), 4),
                      "rmse_mean": round(float(np.mean(rmses)), 4), "exact_rank_runs": exact, "runs": SEEDS})
        print(f"  {n:>6} comparisons: tau {sweep[-1]['kendall_tau_mean']:.3f} (min {sweep[-1]['kendall_tau_min']:.3f}), "
              f"RMSE {sweep[-1]['rmse_mean']:.3f}, exact order in {exact}/{SEEDS}")

    gaps = np.diff(s_true[::-1])
    res = {
        "n_items": N_ITEMS, "s_true": [round(float(x), 4) for x in s_true],
        "smallest_gap": round(float(gaps.min()), 4),
        "fit": {"optimizer": "gradient descent", "lr": 4.0, "steps": 4000, "l2": 1e-4},
        "main": main_run, "sweep": sweep, "guard": guard.report(),
    }
    write_results("bt_synthetic", res)
    if args.figures:
        fig_bt(res, FIGURES / "bt_synthetic.png")


def fig_bt(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    st, sf = np.array(r["s_true"]), np.array(r["main"]["s_fit"])
    lim = (min(st.min(), sf.min()) - 0.3, max(st.max(), sf.max()) + 0.3)
    a.plot(lim, lim, color="#bbbbbb", lw=1)
    a.scatter(st, sf, color="#b7410e", zorder=3)
    for k, (x, y) in enumerate(zip(st, sf)):
        a.annotate(chr(65 + k), (x, y), xytext=(4, -10), textcoords="offset points", fontsize=8, color="#555555")
    a.set_xlabel("hidden score")
    a.set_ylabel("score fitted from outcomes")
    a.set_title(f"{r['main']['comparisons']:,} comparisons: Kendall τ {r['main']['kendall_tau']:.2f}", fontsize=10)
    n = [x["comparisons"] for x in r["sweep"]]
    b.plot(n, [x["kendall_tau_mean"] for x in r["sweep"]], marker="o", color="#b7410e", label="Kendall τ (mean of 20)")
    b.plot(n, [x["exact_rank_runs"] / x["runs"] for x in r["sweep"]], marker="s", color="#555555",
           label="share of runs with the exact order")
    b.set_xscale("log")
    b.set_ylim(0, 1.05)
    b.set_xlabel("comparisons")
    b.set_title("Recovery against the number of comparisons", fontsize=10)
    b.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
