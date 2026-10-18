# mlcourse-24-reinforcement-learning (Python lane)

Python code for module 24, reinforcement learning. See the module overview
one level up: [`../README.md`](../README.md).

## Layout

- `src/common.py`: the laptop guard (MPS watermarks, peak memory, timing), results I/O, the return curve on a step grid
- `src/cartpole.py`: CartPole written by hand from Barto, Sutton and Anderson (1983); one environment and a vectorised copy
- `src/bandits.py`: section 1, ε-greedy, UCB1 and Thompson sampling on a five-armed Bernoulli bandit, 20 seeds
- `src/qlearning.py`: section 2, tabular Q-learning on a 10 × 10 grid with walls, checked against BFS and value iteration
- `src/reinforce.py`: section 3, REINFORCE on CartPole, 3 seeds
- `src/ppo.py`: section 4, PPO on CartPole, 3 seeds; `--lr` and `--no-clip` for the clip comparison
- `src/figures.py`: redraws every figure from `../results/*.json`
- `src/export_explorable.py`: writes the explorable's data block from `../results/*.json`
- `tests/`: CartPole against hand-worked values and the Euler–Lagrange equations; the solutions against the scripts; the results against the budget, known answers and the explorable

## Quickstart

```bash
uv sync
uv run pytest
```
