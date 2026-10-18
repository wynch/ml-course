"""Section 2: train the two models of the grid world, then read what each has learned.

Data: 4,000 training arenas x 25 random actions = 100,000 transitions (seed 1);
500 held-out arenas x 25 = 12,500 transitions for the numbers (seed 2).
Both models: 6,000 Adam steps of 256 transitions, rate 1e-3, on the CPU.

Then, on held-out data:
  - one-hot model: is the top cell the true next cell? (one step)
  - JEPA: prediction loss; the probe's R^2 for the goal offset; module 16's
    collapse meters; the same for the untrained JEPA (the control);
  - both: imagine 8 random actions from 2,000 held-out states and check the
    imagined agent cell against the real one after each step.

    uv run python src/train.py --figures
"""

from __future__ import annotations

import argparse

import common
import numpy as np
import torch

import grid
from grid import N
from models import (collapse_meters, fit_probe, jepa_loss, n_params, r2, random_jepa, save, target_latents,
                    tensors, train_jepa, train_onehot)
import models as M

TRAIN_SEED, TRAIN_ARENAS, HELD_SEED, HELD_ARENAS, WALK = 1, 4000, 2, 500, 25
ROLL_STATES, ROLL_H = 2000, 8


@torch.no_grad()
def rollout_accuracy(models: dict, probes: dict, seed: int = 4) -> dict:
    """Imagined against real agent cell after k = 1..8 random actions, from ROLL_STATES fresh states."""
    rng = np.random.default_rng(seed)
    obs, acts, real, goals, starts = [], [], [], [], []
    for _ in range(ROLL_STATES):
        walls, pos, goal = grid.make_arena(rng)
        a = rng.integers(4, size=ROLL_H)
        obs.append(grid.observe(walls, pos, goal))
        starts.append(pos)
        cells = []
        for k in range(ROLL_H):
            pos = grid.step(walls, pos, int(a[k]))
            cells.append(pos)
        acts.append(a), real.append(cells), goals.append(goal)
    obs = torch.from_numpy(np.stack(obs))
    acts = torch.from_numpy(np.stack(acts))
    real, goals = np.array(real), np.array(goals)[:, None, :]
    out = {}
    for name, m in models.items():
        if name == "onehot":
            cells = m.rollout(obs, acts).numpy()
        else:
            off = probes[name](m.rollout(obs, acts)).numpy()
            cells = np.clip(np.rint(goals - off), 0, N - 1)
        out[name] = [round(float((cells[:, k] == real[:, k]).all(-1).mean()), 4) for k in range(ROLL_H)]
    starts = np.array(starts)
    out["stay_put"] = [round(float((starts == real[:, k]).all(-1).mean()), 4) for k in range(ROLL_H)]
    return out


def train_all(guard: common.Guard | None = None) -> tuple[dict, dict, dict]:
    """Train both models, build the control, fit the probes. Returns models, probes and the numbers."""
    data = grid.collect(TRAIN_ARENAS, WALK, TRAIN_SEED)
    held = grid.collect(HELD_ARENAS, WALK, HELD_SEED)
    if guard:
        guard.tick("collect")
    onehot, log_onehot = train_onehot(data)
    if guard:
        guard.tick("train_onehot")
    jepa, log_jepa = train_jepa(data)
    if guard:
        guard.tick("train_jepa")
    models = {"onehot": onehot, "latent": jepa, "latent_random": random_jepa()}
    probes = {k: fit_probe(models[k], data) for k in ("latent", "latent_random")}

    H = tensors(held)
    with torch.no_grad():
        acc1 = float((onehot(H["obs"], H["action"]).argmax(-1) == H["next_idx"]).float().mean())
        moved = H["next_idx"] != torch.from_numpy(held["pos"][:, 0] * N + held["pos"][:, 1])
        acc1_moved = float((onehot(H["obs"], H["action"]).argmax(-1) == H["next_idx"])[moved].float().mean())
    y = held["goal"] - held["pos"]
    latent = {}
    for k in ("latent", "latent_random"):
        Z = target_latents(models[k], held["obs"])
        with torch.no_grad():
            loss = float(jepa_loss(models[k], H["obs"], H["action"], H["next_obs"]))
        latent[k] = {"pred_loss_held_out": round(loss, 6),
                     "probe_r2_goal_offset": round(r2(probes[k](Z), y), 4),
                     "meters": collapse_meters(Z[:2048])}
    report = {
        "data": {"train_arenas": TRAIN_ARENAS, "held_arenas": HELD_ARENAS, "walk": WALK,
                 "train_transitions": int(len(data["obs"])), "held_transitions": int(len(held["obs"])),
                 "seeds": {"train": TRAIN_SEED, "held": HELD_SEED},
                 "blocked_share_train": round(float((data["pos"] == data["next_pos"]).all(-1).mean()), 4)},
        "settings": {"steps": M.STEPS, "batch": M.BATCH, "lr": M.LR, "latent": M.LATENT, "hidden": M.HIDDEN,
                     "ema": [M.EMA_START, M.EMA_END], "ridge": M.RIDGE, "probe_fit_states": 20000},
        "params": {"onehot": n_params(onehot), "jepa_encoder": n_params(jepa.encoder),
                   "jepa_predictor": n_params(jepa.predictor)},
        "onehot": {"next_cell_acc_held_out": round(acc1, 4), "next_cell_acc_when_moved": round(acc1_moved, 4),
                   "log": log_onehot},
        "jepa": {**latent["latent"], "log": log_jepa},
        "jepa_random": latent["latent_random"],
        "rollout_cell_acc": rollout_accuracy(models, probes),
        "rollout": {"states": ROLL_STATES, "horizon": ROLL_H, "seed": 4},
    }
    return models, probes, report


def what_the_latent_keeps(models: dict) -> dict:
    """Ridge probes for four labels, on held-out observations: what each latent holds, and the raw input.

    Not used by the planners; it explains them. Also: how far one real step moves the latent,
    against how far apart the latents of two different arenas sit.
    """
    from models import Probe

    data = grid.collect(TRAIN_ARENAS, WALK, TRAIN_SEED)
    held = grid.collect(HELD_ARENAS, WALK, HELD_SEED)
    n = 20_000
    tr = {k: v[:n] for k, v in data.items()}
    labels = {"agent_cell": lambda d: d["pos"], "goal_cell": lambda d: d["goal"],
              "goal_offset": lambda d: d["goal"] - d["pos"],
              "wall_count": lambda d: d["obs"][:, :N * N].sum(1, keepdims=True)}
    feats = {"latent": lambda d: target_latents(models["latent"], d["obs"]),
             "latent_random": lambda d: target_latents(models["latent_random"], d["obs"]),
             "raw_observation": lambda d: d["obs"]}
    out = {"probe_r2": {}}
    for f, F in feats.items():
        Ztr, Zte = F(tr), F(held)
        out["probe_r2"][f] = {lab: round(r2(Probe.fit(Ztr, L(tr))(Zte), L(held)), 4) for lab, L in labels.items()}
    moved = ~(held["pos"] == held["next_pos"]).all(-1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(held["obs"]), size=(2, 5000))
    other = held["arena"][i] != held["arena"][j]
    for k in ("latent", "latent_random"):
        z, z2 = target_latents(models[k], held["obs"]), target_latents(models[k], held["next_obs"])
        out[k] = {"move_one_step": round(float(np.linalg.norm(z2 - z, axis=1)[moved].mean()), 5),
                  "between_arenas": round(float(np.linalg.norm(z[i] - z[j], axis=1)[other].mean()), 5)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("train")
    models, probes, report = train_all(guard)
    report["keeps"] = what_the_latent_keeps(models)
    guard.tick("keeps")
    common.ARTIFACTS.mkdir(exist_ok=True)
    save(models, probes, common.ARTIFACTS / "models.pt")
    report["guard"] = guard.report()
    common.write_results("models", report)
    print(f"  one-hot next cell {report['onehot']['next_cell_acc_held_out']:.4f}; "
          f"probe R2 trained {report['jepa']['probe_r2_goal_offset']:.3f}, "
          f"random {report['jepa_random']['probe_r2_goal_offset']:.3f}")
    if args.figures:
        import figures

        figures.models()


if __name__ == "__main__":
    main()
