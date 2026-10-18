"""Section 2: tabular Q-learning on a 10 x 10 grid with two walls.

The agent starts top-left and must reach the goal bottom-right. Four moves;
a move into a wall or off the grid leaves it where it is. Reward -1 per
move, no discount (gamma = 1), so the best value of a cell is minus its
number of moves to the goal, and breadth-first search gives that exactly.

Q-learning (Watkins, 1989), one update per move:
    Q(s, a) <- Q(s, a) + alpha (r + gamma max_a' Q(s', a') - Q(s, a))
with Q(goal, .) = 0, epsilon-greedy moves, and Q starting at 0 everywhere.

    uv run python src/qlearning.py [--figures]
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from common import Guard, write_results

SIZE = 10
START = (0, 0)
GOAL = (9, 9)
# Two walls: column 3 open only at the bottom row, column 6 open only at the top row.
WALLS = frozenset({(r, 3) for r in range(0, 9)} | {(r, 6) for r in range(1, 10)})
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))          # up, right, down, left
ALPHA, EPSILON, GAMMA = 0.5, 0.1, 1.0
EPISODES = 300
SEEDS = 20
STEP_CAP = 2_000                                       # moves per episode before we stop it


def move(cell, a):
    r, c = cell[0] + MOVES[a][0], cell[1] + MOVES[a][1]
    if not (0 <= r < SIZE and 0 <= c < SIZE) or (r, c) in WALLS:
        return cell
    return (r, c)


def bfs_distances(goal=GOAL):
    """Moves to the goal from every free cell; -1 for walls."""
    dist = -np.ones((SIZE, SIZE), dtype=int)
    dist[goal] = 0
    q = deque([goal])
    while q:
        cell = q.popleft()
        for a in range(4):
            nb = move(cell, a)
            if nb != cell and dist[nb] < 0:
                # moves are symmetric on this grid, so a neighbour of a cell can reach it in one move
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return dist


def value_iteration(tol=1e-12):
    """Q* by sweeping Q(s, a) = -1 + max_a' Q(s', a') until nothing changes."""
    Q = np.zeros((SIZE, SIZE, 4))
    while True:
        delta = 0.0
        for r in range(SIZE):
            for c in range(SIZE):
                if (r, c) == GOAL or (r, c) in WALLS:
                    continue
                for a in range(4):
                    nb = move((r, c), a)
                    new = -1.0 + (0.0 if nb == GOAL else GAMMA * Q[nb].max())
                    delta = max(delta, abs(new - Q[r, c, a]))
                    Q[r, c, a] = new
        if delta < tol:
            return Q


def q_update(Q, s, a, r, s2, done, alpha=ALPHA, gamma=GAMMA):
    target = r + (0.0 if done else gamma * Q[s2].max())
    Q[s][a] += alpha * (target - Q[s][a])


def choose(Q, s, rng, eps=EPSILON):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def greedy_path(Q, cap=4 * SIZE * SIZE):
    """Follow argmax Q from the start (first best action on ties). None if it loops."""
    s, path = START, [START]
    while s != GOAL and len(path) <= cap:
        s = move(s, int(np.argmax(Q[s])))
        path.append(s)
    return path if s == GOAL else None


def train(seed: int):
    rng = np.random.default_rng(seed)
    Q = np.zeros((SIZE, SIZE, 4))
    steps = []
    for _ in range(EPISODES):
        s, n = START, 0
        while s != GOAL and n < STEP_CAP:
            a = choose(Q, s, rng)
            s2 = move(s, a)
            q_update(Q, s, a, -1.0, s2, s2 == GOAL)
            s, n = s2, n + 1
        steps.append(n)
    return Q, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("qlearning.py")

    dist = bfs_distances()
    Qstar = value_iteration()
    free = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS and (r, c) != GOAL]
    vi_matches_bfs = all(Qstar[s].max() == -dist[s] for s in free)
    shortest = int(dist[START])
    guard.tick("bfs and value iteration")

    runs = []
    for seed in range(SEEDS):
        Q, steps = train(seed)
        path = greedy_path(Q)
        V = Q.max(axis=2)
        on_path_err = max(abs(V[s] + dist[s]) for s in path[:-1]) if path else None
        optimal_action = np.mean([Qstar[s][int(np.argmax(Q[s]))] == Qstar[s].max() for s in free])
        visited = int(np.sum([np.any(Q[s] != 0) for s in free]))
        runs.append({"seed": seed, "steps": steps, "greedy_path_len": (len(path) - 1) if path else None,
                     "max_abs_value_error_on_greedy_path": round(float(on_path_err), 6) if path else None,
                     "share_cells_greedy_optimal": round(float(optimal_action), 3),
                     "cells_updated": visited, "path": [list(p) for p in path] if path else None,
                     "V": np.round(V, 2).tolist()})
    guard.tick("train")

    steps = np.array([r["steps"] for r in runs])
    last50 = steps[:, -50:].mean(axis=1)
    lens = [r["greedy_path_len"] for r in runs]
    # epsilon-greedy pays for its random moves even once Q is right: a rough floor for the training curve
    for r, ln in zip(runs, lens):
        print(f"seed {r['seed']:2d}: first episode {r['steps'][0]:5d} moves, last-50 mean {np.mean(r['steps'][-50:]):6.1f}, "
              f"greedy path {ln}, greedy optimal in {r['share_cells_greedy_optimal']:.0%} of cells")
    payload = {
        "size": SIZE, "start": list(START), "goal": list(GOAL), "walls": sorted([list(w) for w in WALLS]),
        "alpha": ALPHA, "epsilon": EPSILON, "gamma": GAMMA, "episodes": EPISODES, "seeds": SEEDS,
        "step_cap": STEP_CAP, "reward": -1,
        "bfs_shortest": shortest, "bfs_dist": dist.tolist(), "value_iteration_matches_bfs": bool(vi_matches_bfs),
        "summary": {
            "first_episode_steps_median": float(np.median(steps[:, 0])),
            "first_episode_steps_min": int(steps[:, 0].min()), "first_episode_steps_max": int(steps[:, 0].max()),
            "last50_mean_steps_median": round(float(np.median(last50)), 1),
            "last50_mean_steps_min": round(float(last50.min()), 1),
            "last50_mean_steps_max": round(float(last50.max()), 1),
            "greedy_path_equals_bfs": sum(ln == shortest for ln in lens),
            "share_cells_greedy_optimal_median": float(np.median([r["share_cells_greedy_optimal"] for r in runs])),
            "share_cells_greedy_optimal_min": float(min(r["share_cells_greedy_optimal"] for r in runs)),
            "episodes_to_first_shortest_median": float(np.median([next((i + 1 for i, n in enumerate(r["steps"]) if n == shortest), EPISODES + 1) for r in runs])),
            "capped_episodes": int((steps >= STEP_CAP).sum()),
            "value_error_on_greedy_path_max": max(r["max_abs_value_error_on_greedy_path"] for r in runs if r["path"]),
        },
        "runs": runs,
        "guard": guard.report(),
    }
    print(f"BFS shortest path: {shortest} moves; value iteration matches BFS: {vi_matches_bfs}")
    print(payload["summary"])
    write_results("qlearning", payload)
    if args.figures:
        import figures
        figures.fig_qlearning()


if __name__ == "__main__":
    main()
