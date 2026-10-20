"""Section 1: the two-room world, its data and its test tasks, in numbers.

    uv run python src/world.py --figures
"""

from __future__ import annotations

import argparse

import common
import numpy as np

import plan
import train
import tworoom as W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = common.Guard("world")
    tr, held = train.data()
    guard.tick("collect")
    P = tr["pos"]
    left_right = (P[:, :, 0] < W.WALL_X).any(1) & (P[:, :, 0] > W.WALL_X).any(1)
    moved = np.hypot(*(P[:, 1:] - P[:, :-1]).reshape(-1, 2).T)
    hist, _, _ = np.histogram2d(P[:, :, 1].ravel(), P[:, :, 0].ravel(), bins=32, range=[[0, W.IMG], [0, W.IMG]])
    ts = plan.tasks()
    expert = {s: [W.run_expert(t["start"], t["goal"], plan.BUDGET) for t in v] for s, v in ts.items()}
    guard.tick("tasks")
    out = {
        "geometry": {"img_units": W.IMG, "px": W.PX, "unit_per_px": W.UNIT, "speed": W.SPEED, "radius": W.RADIUS,
                     "wall_x": W.WALL_X, "door": [W.DOOR_Y - W.DOOR_HALF, W.DOOR_Y + W.DOOR_HALF], "reach": W.REACH,
                     "free_x_left": [W.LO, W.WALL_L], "free_x_right": [W.WALL_R, W.HI], "free_y": [W.LO, W.HI],
                     "wall_pixels": int((W.BACKGROUND[0] == 0).sum())},
        "data": {"episodes": int(P.shape[0]), "steps": int(P.shape[1] - 1), "frames": int(P.shape[0] * P.shape[1]),
                 "transitions": int(P.shape[0] * (P.shape[1] - 1)), "held_episodes": int(held["pos"].shape[0]),
                 "crossing_share": round(float(left_right.mean()), 4),
                 "mean_move_units": round(float(moved.mean()), 3), "still_share": round(float((moved < 0.5).mean()), 4),
                 "position_hist_32": hist.astype(int).tolist()},
        "tasks": {s: {"n": len(v), "geodesic_mean": round(float(np.mean([t["geodesic"] for t in v])), 1),
                      "geodesic_min": round(float(np.min([t["geodesic"] for t in v])), 1),
                      "geodesic_max": round(float(np.max([t["geodesic"] for t in v])), 1),
                      "drawn": v[-1]["drawn"],
                      "expert_success": float(np.mean([e["success"] for e in expert[s]])),
                      "expert_steps_mean": round(float(np.mean([e["steps"] for e in expert[s]])), 1),
                      "expert_steps_max": int(np.max([e["steps"] for e in expert[s]]))} for s, v in ts.items()},
        "examples": {"episode_paths": [P[e].round(1).tolist() for e in (0, 1, 2, 3)],
                     "task_cross": ts["cross"][0], "task_same": ts["same"][0],
                     "expert_cross_path": expert["cross"][0]["path"]},
    }
    c = (np.arange(32) + 0.5) * W.IMG / 32
    gy, gx = np.meshgrid(c, c, indexing="ij")
    free = (gy >= W.LO) & (gy <= W.HI) & (((gx >= W.LO) & (gx <= W.WALL_L)) | ((gx >= W.WALL_R) & (gx <= W.HI)))
    out["data"]["empty_bins_share_free_area"] = round(float((hist[free] == 0).mean()), 4)
    out["guard"] = guard.report()
    common.write_results("world", out)
    print(f"  {out['data']['transitions']} transitions, {out['data']['crossing_share']:.1%} of episodes cross the wall")
    if args.figures:
        import figures

        figures.world()


if __name__ == "__main__":
    main()
