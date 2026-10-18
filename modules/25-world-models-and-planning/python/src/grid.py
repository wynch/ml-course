"""The grid world: 8 x 8 cells, walls, one agent, one goal, four actions.

A cell is (row, col); row 0 is the top. The edge of the grid blocks like a
wall. A move into a wall or off the grid leaves the agent where it was. The
agent sees the whole grid as three 8 x 8 one-hot planes: walls, agent, goal.

The world itself is the first five functions, about 40 lines; the rest
collects data and picks test arenas.
"""

from __future__ import annotations

from collections import deque

import numpy as np

N = 8                                                   # the grid is N x N
WALL_P = 0.2                                            # each cell is a wall with this chance
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])    # up, down, left, right
ACTIONS = ("up", "down", "left", "right")
OBS_DIM = 3 * N * N


def make_arena(rng: np.random.Generator) -> tuple[np.ndarray, tuple, tuple]:
    """Walls (N, N) bool, then agent and goal on two different free cells."""
    walls = rng.random((N, N)) < WALL_P
    free = np.argwhere(~walls)
    a, g = rng.choice(len(free), size=2, replace=False)
    return walls, tuple(int(v) for v in free[a]), tuple(int(v) for v in free[g])


def step(walls: np.ndarray, pos: tuple, action: int) -> tuple:
    """The next cell; blocked moves stay put."""
    r, c = pos[0] + MOVES[action][0], pos[1] + MOVES[action][1]
    if 0 <= r < N and 0 <= c < N and not walls[r, c]:
        return (int(r), int(c))
    return pos


def bfs(walls: np.ndarray, goal: tuple) -> np.ndarray:
    """Steps from every cell to the goal; -1 where the goal cannot be reached."""
    dist = np.full((N, N), -1, dtype=np.int64)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for a in range(4):
            nxt = step(walls, cell, a)
            if nxt != cell and dist[nxt] < 0:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def observe(walls: np.ndarray, agent: tuple, goal: tuple) -> np.ndarray:
    """Three one-hot planes, flattened: walls, agent, goal. Shape (192,)."""
    o = np.zeros((3, N, N), dtype=np.float32)
    o[0] = walls
    o[1][agent] = 1
    o[2][goal] = 1
    return o.reshape(-1)


def bfs_action(walls: np.ndarray, pos: tuple, dist: np.ndarray) -> int:
    """An action that lowers the BFS distance by one (the first such, in action order)."""
    for a in range(4):
        nxt = step(walls, pos, a)
        if nxt != pos and dist[nxt] == dist[pos] - 1:
            return a
    raise ValueError("no step lowers the distance: the goal is unreachable or reached")


# ------------------------------------------------------------------ data
def collect(n_arenas: int, steps: int, seed: int) -> dict:
    """Random-policy transitions: n_arenas new arenas, `steps` uniform random actions in each.

    The walk does not stop at the goal: the models learn how the agent moves,
    and the goal is part of the scene. Returns arrays of obs, action, next obs,
    agent cell before and after, goal cell, arena index.
    """
    rng = np.random.default_rng(seed)
    O, A, O2, P, P2, G, E = [], [], [], [], [], [], []
    for e in range(n_arenas):
        walls, pos, goal = make_arena(rng)
        for _ in range(steps):
            a = int(rng.integers(4))
            nxt = step(walls, pos, a)
            O.append(observe(walls, pos, goal)); O2.append(observe(walls, nxt, goal))
            A.append(a); P.append(pos); P2.append(nxt); G.append(goal); E.append(e)
            pos = nxt
    return {"obs": np.stack(O), "action": np.array(A), "next_obs": np.stack(O2),
            "pos": np.array(P), "next_pos": np.array(P2), "goal": np.array(G), "arena": np.array(E)}


def test_arenas(n: int, seed: int, min_dist: int = 5, max_dist: int = 20) -> list[dict]:
    """n arenas whose goal is reachable from the start in min_dist to max_dist steps.

    Arenas that fail the filter are drawn again; how many were drawn is kept,
    so the share the filter removed can be reported.
    """
    rng = np.random.default_rng(seed)
    out, drawn = [], 0
    while len(out) < n:
        walls, start, goal = make_arena(rng)
        drawn += 1
        d = int(bfs(walls, goal)[start])
        if min_dist <= d <= max_dist:
            out.append({"walls": walls, "start": start, "goal": goal, "bfs": d, "drawn": drawn})
    return out
