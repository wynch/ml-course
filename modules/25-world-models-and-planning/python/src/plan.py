"""Section 3: plan in each model by random shooting, and compare with random and BFS.

Random shooting, replanned at every step (model-predictive control):
  1. draw 64 sequences of 8 random actions;
  2. imagine each one in the model, from the current observation;
  3. score each by the sum, over its 8 imagined steps, of the distance to the goal;
  4. take the first action of the lowest score in the real world; repeat.

The distance a planner sees:
  - one-hot model: |row - goal row| + |col - goal col| of the cell it imagines;
  - JEPA: the length of the goal offset the probe reads from the imagined latent.
Both ignore walls, so walls are the model's problem, not the score's.

Agents besides the two planners: BFS (follows a shortest path; the ceiling),
random actions (the floor), the JEPA planner with every weight left random
(the control: the probe and the planner, with nothing learned), and the same
planner with the true world as its model ("perfect"; what this score and this
search can do at best).

    uv run python src/plan.py --figures      # needs artifacts/models.pt from train.py
"""

from __future__ import annotations

import argparse

import common
import numpy as np
import torch

import grid
from grid import N, bfs, bfs_action, observe, step
from models import Probe

SHOTS, HORIZON, BUDGET = 64, 8, 20
TEST_SEED, N_TEST = 3, 300
MIN_DIST, MAX_DIST = 5, 20
BINS = ((5, 6), (7, 8), (9, 10), (11, 20))


# --------------------------------------------------------------- planners
def imagined_distance(kind: str, model, probe: Probe | None, obs: np.ndarray, goal: tuple,
                      actions: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(shots, H) distances and (shots, H, 2) imagined agent cells (floats for the JEPA)."""
    g = np.array(goal, dtype=np.float64)
    if kind == "perfect":
        cells = perfect_rollout(obs, actions.numpy())
        return np.abs(cells - g).sum(-1), cells
    o = torch.from_numpy(obs).unsqueeze(0).expand(len(actions), -1).contiguous()
    if kind == "onehot":
        cells = model.rollout(o, actions).numpy().astype(np.float64)
        return np.abs(cells - g).sum(-1), cells
    off = probe(model.rollout(o, actions)).numpy().astype(np.float64)       # (shots, H, 2)
    return np.sqrt((off ** 2).sum(-1)), g - off


def perfect_rollout(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """The true world as the model: (shots, H) actions -> (shots, H, 2) cells."""
    walls = obs[:N * N].reshape(N, N) > 0
    start = divmod(int(obs[N * N:2 * N * N].argmax()), N)
    out = np.zeros(actions.shape + (2,))
    for s, seq in enumerate(actions):
        pos = start
        for k, a in enumerate(seq):
            pos = step(walls, pos, int(a))
            out[s, k] = pos
    return out


def shoot(kind: str, model, probe, obs, goal, rng: np.random.Generator,
          shots: int = SHOTS, horizon: int = HORIZON) -> tuple[int, np.ndarray, np.ndarray]:
    """One random-shooting decision: (first action, the best sequence, its imagined cells)."""
    actions = torch.from_numpy(rng.integers(4, size=(shots, horizon)))
    dist, cells = imagined_distance(kind, model, probe, obs, goal, actions)
    best = int(np.argmin(dist.sum(1)))
    return int(actions[best, 0]), actions[best].numpy(), cells[best]


def run_episode(agent: str, arena: dict, rng: np.random.Generator, models=None, probes=None,
                keep_imagined: bool = False) -> dict:
    """One episode of at most BUDGET real steps. agent: bfs, random, onehot, latent, latent_random."""
    walls, pos, goal = arena["walls"], arena["start"], arena["goal"]
    dist = bfs(walls, goal)
    path, imagined = [pos], []
    for t in range(BUDGET):
        if agent == "bfs":
            a = bfs_action(walls, pos, dist)
        elif agent == "random":
            a = int(rng.integers(4))
        else:
            kind = agent if agent in ("onehot", "perfect") else "latent"
            a, seq, cells = shoot(kind, (models or {}).get(agent), (probes or {}).get(agent),
                                  observe(walls, pos, goal), goal, rng)
            if keep_imagined:
                imagined.append(np.round(cells, 2).tolist())
        pos = step(walls, pos, a)
        path.append(pos)
        if pos == goal:
            break
    out = {"success": pos == goal, "steps": len(path) - 1, "path": [list(p) for p in path]}
    if keep_imagined:
        out["imagined"] = imagined
    return out


def bin_rate(succ: np.ndarray, m: np.ndarray) -> dict:
    return {"n": int(m.sum()), "success": round(float(succ[m].mean()), 4) if m.any() else None}


AGENTS = ("bfs", "onehot", "latent", "latent_random", "random")


def evaluate(models: dict, probes: dict, arenas: list[dict], keep: int = 12, agents=AGENTS) -> dict:
    """Every agent on every arena. Arena i seeds its own generator, the same for every agent."""
    res = {}
    for agent in agents:
        succ, steps, episodes = [], [], []
        for i, ar in enumerate(arenas):
            ep = run_episode(agent, ar, np.random.default_rng(1000 + i), models, probes,
                             keep_imagined=i < keep and agent in ("onehot", "latent"))
            succ.append(ep["success"])
            steps.append(ep["steps"])
            if i < keep:
                episodes.append(ep)
        succ = np.array(succ)
        steps = np.array(steps)
        d = np.array([a["bfs"] for a in arenas])
        res[agent] = {
            "success": round(float(succ.mean()), 4),
            "n_success": int(succ.sum()),
            "mean_steps_when_success": round(float(steps[succ].mean()), 2) if succ.any() else None,
            "mean_excess_steps_when_success": round(float((steps - d)[succ].mean()), 2) if succ.any() else None,
            "by_distance": {f"{lo}-{hi}": bin_rate(succ, (d >= lo) & (d <= hi)) for lo, hi in BINS},
            "episodes": episodes,
        }
    return res


def arena_summary(arenas: list[dict]) -> dict:
    d = np.array([a["bfs"] for a in arenas])
    return {"n": len(arenas), "seed": TEST_SEED, "min_dist": MIN_DIST, "max_dist": MAX_DIST,
            "drawn": arenas[-1]["drawn"], "bfs_mean": round(float(d.mean()), 2),
            "bfs_median": float(np.median(d)), "bfs_max": int(d.max()),
            "wall_share": round(float(np.mean([a["walls"].mean() for a in arenas])), 4),
            "first": [{"walls": a["walls"].astype(int).tolist(), "start": list(a["start"]),
                       "goal": list(a["goal"]), "bfs": a["bfs"]} for a in arenas[:12]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    from models import load

    guard = common.Guard("plan")
    models, probes = load(common.ARTIFACTS / "models.pt")
    arenas = grid.test_arenas(N_TEST, TEST_SEED, MIN_DIST, MAX_DIST)
    guard.tick("arenas")
    res = evaluate(models, probes, arenas)
    # The true world as the model: what random shooting with this score can do at best.
    res.update(evaluate(models, probes, arenas, agents=("perfect",)))
    guard.tick("plan")
    for a in AGENTS + ("perfect",):
        print(f"  {a:14s} success {res[a]['success']:.3f}")
    common.write_results("plan", {"shots": SHOTS, "horizon": HORIZON, "budget": BUDGET,
                                  "arenas": arena_summary(arenas), "agents": res, "guard": guard.report()})
    if args.figures:
        import figures

        figures.plan()


if __name__ == "__main__":
    main()
