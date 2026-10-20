"""The two-room world: a point agent, a wall with one door, frames of 64 x 64 pixels.

A port of stable-worldmodel's TwoRoom (``swm/TwoRoom-v1``, default settings)
that works on many agents at once. The state lives in the library's own units,
a 224 x 224 frame, so section 5 can check this port against the library step
for step; only the picture is drawn at 64 x 64 (3.5 library units a pixel).

- The agent is a point (x, y), x to the right, y down. An action is a 2-D
  velocity in [-1, 1]^2; one step moves the agent by 5 units times it.
- Borders keep the agent 21 units from each edge (border 14 plus radius 7).
- A vertical wall at x = 112, 10 units thick, splits two rooms. The door is
  the stretch y in [35, 63]. A move that would bring the agent within its
  radius of the wall, with the new y outside the door (plus 1.75 slack),
  leaves it 0.5 short of that line.
- The goal is reached when the agent is within 16 units of it.

The world itself is the first four functions; the rest collects data, draws
test tasks and measures distances.
"""

from __future__ import annotations

import numpy as np
import torch

IMG = 224.0                      # library units across the frame
PX = 64                          # pixels across the frame we draw
UNIT = IMG / PX                  # 3.5 library units per pixel
BORDER, RADIUS, SPEED = 14.0, 7.0, 5.0
LO, HI = BORDER + RADIUS, IMG - BORDER - RADIUS          # 21, 203: where the agent's centre can be
WALL_X, WALL_HALF = 112.0, 5.0
WALL_L, WALL_R = WALL_X - WALL_HALF - RADIUS, WALL_X + WALL_HALF + RADIUS   # 100, 124
DOOR_Y, DOOR_HALF, DOOR_SLACK = 49.0, 14.0, 1.75
DOOR_LO, DOOR_HI = DOOR_Y - DOOR_HALF - DOOR_SLACK, DOOR_Y + DOOR_HALF + DOOR_SLACK   # 33.25, 64.75
REACH = 16.0                     # success radius, library units
ACT_DIM = 2


def step(pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(N, 2) positions and actions -> (N, 2) next positions, float32, as the library computes them."""
    pos = np.asarray(pos, dtype=np.float32)
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    nxt = pos + a * np.float32(SPEED)
    x = np.clip(nxt[:, 0], LO, HI)
    y = np.clip(nxt[:, 1], LO, HI)
    in_door = (y >= DOOR_LO) & (y <= DOOR_HI)
    left = pos[:, 0] < WALL_X
    x = np.where(left & (x > WALL_L) & ~in_door, WALL_L - 0.5, x)
    x = np.where(~left & (x < WALL_R) & ~in_door, WALL_R + 0.5, x)
    return np.stack([x, y], 1).astype(np.float32)


def _background() -> torch.Tensor:
    """(3, 64, 64): white rooms, black wall and border lines, sampled at each pixel's centre."""
    u = (torch.arange(PX, dtype=torch.float32) + 0.5) * UNIT - 0.5        # pixel centres, library units
    gy, gx = torch.meshgrid(u, u, indexing="ij")
    wall = (gx >= WALL_X - WALL_HALF) & (gx <= WALL_X + WALL_HALF) & ~((gy >= DOOR_Y - DOOR_HALF) & (gy <= DOOR_Y + DOOR_HALF))
    for g in (gx, gy):                                                    # border lines, 4 units thick
        wall |= ((g >= BORDER - 4) & (g < BORDER)) | ((g >= IMG - BORDER) & (g < IMG - BORDER + 4))
    return (~wall).float().expand(3, PX, PX).clone()


BACKGROUND = _background()
_U = (torch.arange(PX, dtype=torch.float32) + 0.5) * UNIT - 0.5
RED = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)


def render(pos) -> torch.Tensor:
    """(N, 2) positions -> (N, 3, 64, 64) frames in [0, 1]: the agent a red Gaussian dot (sd 7 units) over the rooms."""
    p = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
    dx = _U.view(1, 1, PX) - p[:, 0].view(-1, 1, 1)
    dy = _U.view(1, PX, 1) - p[:, 1].view(-1, 1, 1)
    a = torch.exp(-(dx ** 2 + dy ** 2) / (2 * RADIUS ** 2)).unsqueeze(1)  # (N, 1, 64, 64)
    return BACKGROUND.unsqueeze(0) * (1 - a) + RED * a


def geodesic(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Approximate shortest walk length, library units: straight within a room, through the door between rooms.

    Between rooms, the walk crosses x = 112 at some y in the door; we take the best of 57 such points.
    """
    p, g = np.atleast_2d(p).astype(np.float64), np.atleast_2d(g).astype(np.float64)
    same = (p[:, 0] < WALL_X) == (g[:, 0] < WALL_X)
    yd = np.linspace(DOOR_Y - DOOR_HALF + RADIUS, DOOR_Y + DOOR_HALF - RADIUS, 57)    # the agent must fit
    via = (np.hypot(p[:, :1] - WALL_X, p[:, 1:2] - yd) + np.hypot(g[:, :1] - WALL_X, g[:, 1:2] - yd)).min(1)
    return np.where(same, np.hypot(*(p - g).T), via)


# ------------------------------------------------------------------ policies
def expert_action(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Full speed towards the goal, by way of the door when the goal is in the other room. (N, 2)."""
    pos, goal = np.asarray(pos, np.float64), np.asarray(goal, np.float64)
    way = goal.copy()
    cross = (pos[:, 0] < WALL_X) != (goal[:, 0] < WALL_X)
    side = np.where(pos[:, 0] < WALL_X, -1.0, 1.0)
    near_door = (np.abs(pos[:, 1] - DOOR_Y) < 6) & (np.abs(pos[:, 0] - WALL_X) < 30)
    way[cross] = np.stack([WALL_X + side[cross] * 22, np.full(cross.sum(), DOOR_Y)], 1)       # line up with the door
    through = cross & near_door
    way[through] = np.stack([WALL_X - side[through] * 30, np.full(through.sum(), DOOR_Y)], 1)  # walk through it
    d = way - pos
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return (d / np.maximum(n, 1e-6) * np.minimum(1.0, n / SPEED)).astype(np.float32)


def sample_free(rng: np.random.Generator, n: int, side: np.ndarray | None = None) -> np.ndarray:
    """n points where the agent may stand; side -1 = left room, +1 = right room, None = either."""
    if side is None:
        side = rng.choice([-1.0, 1.0], size=n)
    x_left = rng.uniform(LO, WALL_L - 0.5, n)
    x = np.where(side < 0, x_left, x_left + (WALL_R + 0.5 - LO))           # same width on the right
    return np.stack([x, rng.uniform(LO, HI, n)], 1).astype(np.float32)


# ---------------------------------------------------------------------- data
def collect(n_episodes: int, steps: int, seed: int, noise: float = 0.5, wander_share: float = 0.3) -> dict:
    """Trajectories of a noisy expert, the way LeWM's TwoRoom data was made (a noisy heuristic).

    Each episode starts at a random free point and walks `steps` steps. 70% of
    episodes follow the expert towards random goals (a new goal each time one
    is reached), with Gaussian noise of sd `noise` on each action; 30% wander:
    an action that drifts slowly (each step keeps 90% of the last one). Returns
    positions (E, steps + 1, 2) and actions (E, steps, 2), float32.
    """
    rng = np.random.default_rng(seed)
    pos = sample_free(rng, n_episodes)
    goal = sample_free(rng, n_episodes)
    wander = rng.random(n_episodes) < wander_share
    drift = rng.uniform(-1, 1, (n_episodes, 2)).astype(np.float32)
    P, A = [pos], []
    for _ in range(steps):
        reached = np.hypot(*(pos - goal).T) < REACH
        if reached.any():
            goal[reached] = sample_free(rng, int(reached.sum()))
        drift = 0.9 * drift + 0.1 * rng.uniform(-1, 1, (n_episodes, 2)) * 3
        a = expert_action(pos, goal) + noise * rng.standard_normal((n_episodes, 2))
        a = np.where(wander[:, None], drift, a)
        a = np.clip(a, -1, 1).astype(np.float32)
        pos = step(pos, a)
        P.append(pos)
        A.append(a)
    return {"pos": np.stack(P, 1), "action": np.stack(A, 1), "seed": seed}


def test_tasks(n: int, seed: int, cross: bool, min_geo: float = 60.0) -> list[dict]:
    """n (start, goal) pairs at least `min_geo` units apart along the walk; in different rooms if `cross`."""
    rng = np.random.default_rng(seed)
    out, drawn = [], 0
    while len(out) < n:
        s = sample_free(rng, 1)
        side = -np.sign(s[0, 0] - WALL_X) if cross else np.sign(s[0, 0] - WALL_X)
        g = sample_free(rng, 1, side=np.array([side]))
        drawn += 1
        d = float(geodesic(s, g)[0])
        if d >= min_geo:
            out.append({"start": s[0].tolist(), "goal": g[0].tolist(), "geodesic": round(d, 2), "drawn": drawn})
    return out


def run_expert(start, goal, budget: int) -> dict:
    """The noise-free expert on one task: the ceiling."""
    pos, g = np.array([start], np.float32), np.array([goal], np.float32)
    path = [pos[0].tolist()]
    for _ in range(budget):
        pos = step(pos, expert_action(pos, g))
        path.append(pos[0].tolist())
        if np.hypot(*(pos[0] - g[0])) < REACH:
            break
    return {"success": bool(np.hypot(*(pos[0] - g[0])) < REACH), "steps": len(path) - 1, "path": path}
