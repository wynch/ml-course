"""Section 1: the grid world, what its arenas look like, and the data the models learn from.

No model here: arena statistics, the test filter, and the random-walk data.

    uv run python src/world.py --figures
"""

from __future__ import annotations

import argparse

import common
import numpy as np

import grid
import plan
import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("world")

    # 10,000 raw arenas, as make_arena draws them, before any filter.
    rng = np.random.default_rng(0)
    walls_share, reach, dists, free = [], [], [], []
    for _ in range(10_000):
        walls, start, goal = grid.make_arena(rng)
        d = int(grid.bfs(walls, goal)[start])
        walls_share.append(walls.mean())
        free.append(int((~walls).sum()))
        reach.append(d >= 0)
        if d >= 0:
            dists.append(d)
    dists = np.array(dists)
    hist = np.bincount(dists, minlength=21)[:21]
    guard.tick("raw_arenas")

    tests = grid.test_arenas(plan.N_TEST, plan.TEST_SEED, plan.MIN_DIST, plan.MAX_DIST)
    td = np.array([a["bfs"] for a in tests])
    data = grid.collect(train.TRAIN_ARENAS, train.WALK, train.TRAIN_SEED)
    blocked = (data["pos"] == data["next_pos"]).all(-1)
    on_goal = (data["pos"] == data["goal"]).all(-1)
    guard.tick("data")

    out = {
        "grid": grid.N, "wall_p": grid.WALL_P, "actions": list(grid.ACTIONS), "obs_dim": grid.OBS_DIM,
        "raw_arenas": {
            "n": 10_000, "seed": 0,
            "wall_share_mean": round(float(np.mean(walls_share)), 4),
            "free_cells_mean": round(float(np.mean(free)), 2),
            "goal_reachable_share": round(float(np.mean(reach)), 4),
            "bfs_mean_when_reachable": round(float(dists.mean()), 2),
            "bfs_max": int(dists.max()),
            "bfs_hist_0_20": hist.tolist(),
            "share_5_to_20": round(float(((dists >= 5) & (dists <= 20)).sum() / len(reach)), 4),
        },
        "test_arenas": {
            "n": len(tests), "seed": plan.TEST_SEED, "min_dist": plan.MIN_DIST, "max_dist": plan.MAX_DIST,
            "drawn": tests[-1]["drawn"], "bfs_mean": round(float(td.mean()), 2),
            "bfs_hist": np.bincount(td, minlength=21)[:21].tolist(),
            "examples": [{"walls": a["walls"].astype(int).tolist(), "start": list(a["start"]),
                          "goal": list(a["goal"]), "bfs": a["bfs"],
                          "dist_map": grid.bfs(a["walls"], a["goal"]).tolist()} for a in tests[:6]],
        },
        "train_data": {
            "arenas": train.TRAIN_ARENAS, "walk": train.WALK, "seed": train.TRAIN_SEED,
            "transitions": int(len(blocked)),
            "blocked_share": round(float(blocked.mean()), 4),
            "on_goal_share": round(float(on_goal.mean()), 4),
            "action_share": np.bincount(data["action"], minlength=4).tolist(),
        },
        "guard": guard.report(),
    }
    common.write_results("world", out)
    r = out["raw_arenas"]
    print(f"  walls {r['wall_share_mean']:.3f}, goal reachable {r['goal_reachable_share']:.3f}, "
          f"test arenas {len(tests)} of {tests[-1]['drawn']} drawn")
    if args.figures:
        import figures

        figures.world()


if __name__ == "__main__":
    main()
