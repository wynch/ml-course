"""Section 3: REINFORCE on the hand-written CartPole, 3 seeds.

The policy is a small network that maps the 4 numbers of the state to the
probability of pushing right. After each episode, every action's log
probability is pushed up in proportion to the return that followed it
(Williams, 1992):

    loss = - sum_t log pi(a_t | s_t) * (G_t - mean G) / std G,    G_t = sum_{k>=t} gamma^(k-t) r_k

Subtracting the episode's mean return and dividing by its spread is a common
trick that makes the step size steady. It is not a true baseline: the mean
and spread come from the same episode's actions, so the estimate is biased.
Since G_t shrinks with t inside an episode, it pushes up the early actions
and pushes down the late ones, whatever the episode's length.

    uv run python src/reinforce.py [--figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from common import Guard, curve_on_grid, write_results
import torch  # noqa: E402  (after common: the MPS guard)

import cartpole as cp

SEEDS = (0, 1, 2)
BUDGET = 200_000            # environment steps per seed
GAMMA = 0.99
LR = 1e-2
HIDDEN = 64
GRID = list(range(5_000, BUDGET + 1, 5_000))
EVAL_EPISODES = 20


class Policy(torch.nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2))

    def forward(self, s):
        return torch.distributions.Categorical(logits=self.net(s))


def returns_to_go(rewards, gamma=GAMMA):
    G, out = 0.0, np.zeros(len(rewards))
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def reinforce_loss(logp: torch.Tensor, G: np.ndarray) -> torch.Tensor:
    g = torch.as_tensor(G, dtype=torch.float32)
    g = (g - g.mean()) / (g.std() + 1e-8) if len(g) > 1 else g * 0
    return -(logp * g).sum()


def greedy(policy):
    def act(s):
        with torch.no_grad():
            return int(policy.net(torch.as_tensor(s, dtype=torch.float32)).argmax())
    return act


def evaluate(policy, seed):
    rng = np.random.default_rng(10_000 + seed)
    return [cp.run_episode(greedy(policy), rng)[0] for _ in range(EVAL_EPISODES)]


def train(seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    env = cp.CartPole(rng)
    total, ends, rets = 0, [], []
    while total < BUDGET:
        s = env.reset()
        logps, rewards = [], []
        while True:
            dist = policy(torch.as_tensor(s, dtype=torch.float32))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            s, r, term, trunc = env.step(int(a))
            rewards.append(r)
            total += 1
            if term or trunc or total >= BUDGET:
                break
        if not (term or trunc):
            break                                   # budget cut the episode short; it is not counted
        loss = reinforce_loss(torch.stack(logps), returns_to_go(rewards))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ends.append(total)
        rets.append(sum(rewards))
    return policy, ends, rets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    guard = Guard("reinforce.py")
    runs = []
    for seed in SEEDS:
        policy, ends, rets = train(seed)
        ev = evaluate(policy, seed)
        first500 = next((e for e, r in zip(ends, rets) if r >= cp.MAX_STEPS), None)
        _, states = cp.run_episode(greedy(policy), np.random.default_rng(20_000), record=True)
        runs.append({"seed": seed, "episodes": len(rets), "ends": ends, "returns": rets,
                     "curve": curve_on_grid(ends, rets, GRID), "eval_greedy": ev,
                     "eval_greedy_mean": float(np.mean(ev)), "first_500_at_step": first500,
                     "replay_len": len(states) - 1,
                     "replay": np.round(states, 4).tolist() if seed == 0 else None})
        guard.tick(f"seed {seed}")
        print(f"seed {seed}: {len(rets)} episodes, last-20 mean {np.mean(rets[-20:]):.1f}, first 500 at step {first500}, "
              f"greedy eval mean {np.mean(ev):.1f} over {EVAL_EPISODES}")
    payload = {"seeds": list(SEEDS), "budget_steps": BUDGET, "gamma": GAMMA, "lr": LR, "hidden": HIDDEN,
               "optimiser": "Adam", "grid": GRID, "curve_window": 20, "eval_episodes": EVAL_EPISODES,
               "runs": runs, "guard": guard.report()}
    write_results("reinforce", payload)
    if args.figures:
        import figures
        figures.fig_cartpole()


if __name__ == "__main__":
    main()
