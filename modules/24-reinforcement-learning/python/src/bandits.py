"""Section 1: a five-armed Bernoulli bandit; epsilon-greedy, UCB1 and Thompson sampling.

Each arm pays 1 with its own probability and 0 otherwise. The rule sees only
what it pulled and what came back. Regret after t pulls is the pseudo-regret
sum over pulls of (best mean - mean of the arm pulled): what the rule lost
against always pulling the best arm, without the noise of the coin flips.

20 seeds; each seed shuffles which arm holds which probability and draws its
own coin flips. All three rules (and a random baseline) see the same coin
flips: reward of arm a at pull t is 1 if u[t, a] < p[a], with one shared u
per seed.

    uv run python src/bandits.py [--figures]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from common import Guard, write_results

MEANS = (0.2, 0.4, 0.5, 0.6, 0.7)
PULLS = 10_000
SEEDS = 20
EPSILON = 0.1
RULES = ("random", "egreedy", "ucb", "thompson")
GRID = sorted(set(range(1, 101)) | set(range(100, PULLS + 1, 25)))   # pulls at which regret is stored


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax, ties broken at random."""
    noise = rng.random(values.shape) * 1e-9
    return np.argmax(values + noise * (values == values.max(axis=1, keepdims=True)), axis=1)


def egreedy_choose(counts, sums, t, rng, eps=EPSILON):
    """With probability eps a random arm; else the arm with the best average so far (0 for untried)."""
    n, k = counts.shape
    est = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    greedy = argmax_random(est, rng)
    explore = rng.random(n) < eps
    return np.where(explore, rng.integers(0, k, size=n), greedy)


def ucb_choose(counts, sums, t, rng):
    """UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): try each arm once, then
    the arm with the largest mean + sqrt(2 ln t / n). t counts pulls so far, from 1."""
    n, k = counts.shape
    if t <= k:
        return np.full(n, t - 1)
    mean = sums / counts
    bonus = np.sqrt(2.0 * math.log(t) / counts)
    return argmax_random(mean + bonus, rng)


def thompson_choose(counts, sums, t, rng):
    """Thompson sampling with a Beta(1, 1) prior: draw each arm's p from its
    Beta(1 + wins, 1 + losses) and pull the arm whose draw is highest."""
    draws = rng.beta(1.0 + sums, 1.0 + counts - sums)
    return argmax_random(draws, rng)


def random_choose(counts, sums, t, rng):
    return rng.integers(0, counts.shape[1], size=counts.shape[0])


CHOOSE = {"random": random_choose, "egreedy": egreedy_choose, "ucb": ucb_choose, "thompson": thompson_choose}


def kl_bernoulli(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def run(rule: str, probs: np.ndarray, u: np.ndarray, rng: np.random.Generator, pulls: int = PULLS):
    """probs (seeds, k); u (pulls, seeds, k). Returns regret at GRID, pull counts, last-1000 best-arm share."""
    n, k = probs.shape
    counts = np.zeros((n, k))
    sums = np.zeros((n, k))
    best = probs.max(axis=1)
    best_arm = probs.argmax(axis=1)
    regret = np.zeros(n)
    grid = set(g for g in GRID if g <= pulls)
    out, late_best = [], np.zeros(n)
    rows = np.arange(n)
    for t in range(1, pulls + 1):
        a = CHOOSE[rule](counts, sums, t, rng)
        r = (u[t - 1, rows, a] < probs[rows, a]).astype(float)
        counts[rows, a] += 1
        sums[rows, a] += r
        regret += best - probs[rows, a]
        if t > pulls - 1000:
            late_best += a == best_arm
        if t in grid:
            out.append(regret.copy())
    return np.array(out).T, counts, late_best / 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    guard = Guard("bandits.py")

    k = len(MEANS)
    probs = np.zeros((SEEDS, k))
    u = np.zeros((PULLS, SEEDS, k))
    for s in range(SEEDS):
        g = np.random.default_rng(1000 + s)
        probs[s] = g.permutation(MEANS)
        u[:, s, :] = g.random((PULLS, k))

    best = max(MEANS)
    gaps = [best - p for p in MEANS if p < best]
    lai_robbins = sum((best - p) / kl_bernoulli(p, best) for p in MEANS if p < best)
    random_slope = best - float(np.mean(MEANS))

    rules = {}
    for rule in RULES:
        rng = np.random.default_rng(7)          # the rule's own coin, apart from the arms' coins
        reg, counts, late = run(rule, probs, u, rng)
        final = reg[:, -1]
        rules[rule] = {
            "regret": np.round(reg, 3).tolist(),             # (seeds, len(GRID))
            "final_regret": np.round(final, 2).tolist(),
            "final_median": round(float(np.median(final)), 1),
            "final_min": round(float(final.min()), 1),
            "final_max": round(float(final.max()), 1),
            "best_arm_share_last_1000": np.round(late, 3).tolist(),
            "best_arm_share_last_1000_median": round(float(np.median(late)), 3),
            "pulls_per_mean": {f"{p:g}": int(np.median([counts[s, list(probs[s]).index(p)] for s in range(SEEDS)]))
                               for p in MEANS},
        }
        guard.tick(rule)
        print(f"{rule:9s} regret after {PULLS:,} pulls: median {rules[rule]['final_median']}, "
              f"range {rules[rule]['final_min']}-{rules[rule]['final_max']}; "
              f"best arm in last 1000: {rules[rule]['best_arm_share_last_1000_median']:.1%}")

    payload = {
        "means": list(MEANS), "pulls": PULLS, "seeds": SEEDS, "epsilon": EPSILON, "grid": GRID,
        "regret_definition": "sum over pulls of (best mean - mean of the arm pulled)",
        "random_expected_slope": round(random_slope, 4),
        "egreedy_expected_late_slope": round(EPSILON * random_slope, 4),
        "lai_robbins_constant": round(lai_robbins, 3),
        "lai_robbins_at_pulls": round(lai_robbins * math.log(PULLS), 1),
        "gaps": gaps,
        "rules": rules,
        "guard": guard.report(),
    }
    write_results("bandits", payload)
    if args.figures:
        import figures
        figures.fig_bandits()


if __name__ == "__main__":
    main()
