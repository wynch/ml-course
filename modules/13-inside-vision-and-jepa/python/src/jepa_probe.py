"""Section 4: what a JEPA latent holds.

Freeze gridjepa's encoder, encode 10,000 arena frames, and ask a linear probe
for three things: the agent's position in the arena, the 8 tiles around it,
and the number of steps of the shortest path to the goal (breadth-first
search on the arena, capped at 30). Compare against three random encoders of
the same shape, a probe on shuffled labels, and a probe on the raw window.

    uv run python src/jepa_probe.py --figures
"""

from __future__ import annotations

import argparse

import common  # noqa: F401  (MPS guard before torch)
import numpy as np
import torch

from common import Guard, write_results
from grid_encoder import (adjacency_labels, encode, load_random, load_states, load_trained,
                          logreg_fit_predict, r2_score, ridge_fit, ridge_predict,
                          split_by_episode)

CAP = 30


def ridge_r2(X, Y, tr, te):
    W = ridge_fit(X[tr], Y[tr])
    pred = ridge_predict(W, X[te])
    return r2_score(pred, Y[te]), pred


def adjacency_acc(X, adj, tr, te) -> float:
    return float(np.mean([(logreg_fit_predict(X[tr], adj[tr, j], X[te], n_classes=10, seed=0)
                           == adj[te, j]).mean() for j in range(8)]))


def path_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = raw.astype(np.float64)
    unreachable = ~np.isfinite(raw)
    capped = unreachable | (raw >= CAP)
    y = np.where(unreachable, CAP, np.minimum(raw, CAP)).astype(np.float32)
    return y, capped, unreachable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("jepa_probe")
    torch.set_num_threads(4)
    st = load_states()
    raw = st["goal_dist"]
    y, capped, unreachable = path_labels(raw)
    tr, te = split_by_episode(st["ep"], seed=0)
    Y = y[:, None]
    pos = st["pos"].astype(np.float32)
    adj = adjacency_labels(st["windows"])

    encoders = {"trained": load_trained()}
    for s in (0, 1, 2):
        encoders[f"random_{s}"] = load_random(s)
    feats = {n: encode(e, st["windows"], st["scalars"]) for n, e in encoders.items()}
    onehot = np.eye(10, dtype=np.float32)[st["windows"].astype(np.int64)].reshape(len(y), -1)
    feats["raw_window"] = np.concatenate([onehot, st["scalars"]], axis=1)
    guard.tick("encode")

    res, preds = {}, {}
    for name, X in feats.items():
        r2_path, p = ridge_r2(X, Y, tr, te)
        r2_pos, _ = ridge_r2(X, pos, tr, te)
        res[name] = {"path_r2": round(r2_path, 4), "position_r2": round(r2_pos, 4),
                     "adjacency_acc": round(adjacency_acc(X, adj, tr, te), 4)}
        preds[name] = p[:, 0]
        print(name, res[name], flush=True)
    perm = np.random.default_rng(0).permutation(len(y))
    r2_shuf, _ = ridge_r2(feats["trained"], Y[perm], tr, te)
    guard.tick("probes")

    _, inv = np.unique(st["ep"], return_inverse=True)
    ep_mean = (np.bincount(inv, weights=y) / np.bincount(inv))[inv]     # each frame's episode mean
    rnd = [res[f"random_{s}"]["path_r2"] for s in (0, 1, 2)]
    trained = res["trained"]["path_r2"]
    # 300 held-out frames for the scatter in the explorable and the figure.
    show = np.random.default_rng(1).choice(len(te), size=300, replace=False)
    write_results("jepa_probe", {
        "encoder": "gridjepa h1: EMA target + stop-gradient, arena 12, window 15, latent 128, 10,000 training steps",
        "n_states": int(len(y)), "n_episodes": int(len(np.unique(st["ep"]))),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "cap": CAP,
        "capped_share": round(float(capped.mean()), 4),
        "unreachable_share": round(float(unreachable.mean()), 4),
        "label_mean": round(float(y.mean()), 4),
        "label_median": float(np.median(y)),
        "on_goal_share": round(float((raw == 0).mean()), 4),
        "n_train_episodes": int(len(np.unique(st["ep"][tr]))),
        "n_test_episodes": int(len(np.unique(st["ep"][te]))),
        # Share of the label's variance that lies between episodes (arenas).
        "between_episode_variance_share": round(float(np.var(ep_mean) / np.var(y)), 4),
        "probes": res,
        "random_path_r2_range": [min(rnd), max(rnd)],
        "trained_minus_best_random": round(trained - max(rnd), 4),
        "trained_minus_mean_random": round(trained - float(np.mean(rnd)), 4),
        "shuffled_label_r2": round(r2_shuf, 4),
        "label_histogram": np.bincount(y.astype(int), minlength=CAP + 1).tolist(),
        "scatter": {"true": y[te][show].round(0).tolist(),
                    **{n: preds[n][show].round(2).tolist() for n in ("trained", "random_0", "raw_window")}},
        "device": "cpu",
        "guard": guard.report(),
    })
    print(f"path R2: trained {trained}, random {rnd}, shuffled {r2_shuf:.4f}")
    if args.figures:
        import figures

        figures.fig_jepa_probe()


if __name__ == "__main__":
    main()
