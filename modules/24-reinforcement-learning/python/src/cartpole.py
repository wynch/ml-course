"""CartPole, written by hand: the dynamics of Barto, Sutton and Anderson (1983).

A pole is hinged to a cart that runs on a frictionless track. Each step the
agent pushes the cart left or right with a fixed force; the reward is 1 per
step, and the episode ends when the pole leans past 12 degrees, the cart
leaves the track, or 500 steps pass.

Equations of motion, with the two friction terms of the 1983 paper set to 0
(theta is the pole's angle from upright, positive when the top leans to +x;
F is the force on the cart):

    temp     = (F + m l theta_dot^2 sin theta) / (M + m)
    theta_dd = (g sin theta - cos theta * temp) / (l (4/3 - m cos^2 theta / (M + m)))
    x_dd     = temp - m l theta_dd cos theta / (M + m)

l is half the pole's length. One explicit Euler step of tau = 0.02 s:
x += tau x_dot; x_dot += tau x_dd; theta += tau theta_dot; theta_dot += tau theta_dd.

Sources: A. G. Barto, R. S. Sutton, C. W. Anderson, "Neuronlike adaptive
elements that can solve difficult learning control problems", IEEE Trans.
SMC 13(5), 1983, equations (23) and (24) and their constants; the classic
gym implementation (gym/envs/classic_control/cartpole.py, CartPole-v1) for
the 500-step limit, the 12-degree and 2.4 m limits, the Euler order and the
start state drawn uniformly from [-0.05, 0.05]^4.
"""

from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.8           # m/s^2
MASS_CART = 1.0         # kg
MASS_POLE = 0.1         # kg
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LENGTH = 0.5       # m, half the pole's length
POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH
FORCE_MAG = 10.0        # N, action 1 pushes +x, action 0 pushes -x
TAU = 0.02              # s per step
THETA_LIMIT = 12 * 2 * math.pi / 360    # rad, as gym writes it
X_LIMIT = 2.4           # m
MAX_STEPS = 500         # CartPole-v1


def derivatives(state: np.ndarray, force: np.ndarray | float):
    """(x_dd, theta_dd) for states of shape (..., 4) = (x, x_dot, theta, theta_dot)."""
    theta, theta_dot = state[..., 2], state[..., 3]
    cos, sin = np.cos(theta), np.sin(theta)
    temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    theta_dd = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * cos**2 / TOTAL_MASS))
    x_dd = temp - POLE_MASS_LENGTH * theta_dd * cos / TOTAL_MASS
    return x_dd, theta_dd


def step_state(state: np.ndarray, action: np.ndarray | int) -> np.ndarray:
    """One Euler step for states of shape (..., 4) and actions in {0, 1}."""
    force = np.where(np.asarray(action) == 1, FORCE_MAG, -FORCE_MAG)
    x_dd, theta_dd = derivatives(state, force)
    x, x_dot, theta, theta_dot = (state[..., i] for i in range(4))
    return np.stack([x + TAU * x_dot, x_dot + TAU * x_dd,
                     theta + TAU * theta_dot, theta_dot + TAU * theta_dd], axis=-1)


def failed(state: np.ndarray) -> np.ndarray:
    return (np.abs(state[..., 0]) > X_LIMIT) | (np.abs(state[..., 2]) > THETA_LIMIT)


class CartPole:
    """One environment. ``reset`` takes a numpy Generator, so every episode is seeded."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.state = None
        self.t = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, action: int):
        """Returns (state, reward, terminated, truncated), as gymnasium does."""
        self.state = step_state(self.state, action)
        self.t += 1
        terminated = bool(failed(self.state))
        truncated = (not terminated) and self.t >= MAX_STEPS
        return self.state.copy(), 1.0, terminated, truncated


class VecCartPole:
    """n environments stepped together; a finished one resets at once (its last state is returned apart)."""

    def __init__(self, n: int, rng: np.random.Generator):
        self.n, self.rng = n, rng
        self.state = rng.uniform(-0.05, 0.05, size=(n, 4))
        self.t = np.zeros(n, dtype=np.int64)

    def step(self, actions: np.ndarray):
        nxt = step_state(self.state, actions)
        self.t += 1
        terminated = failed(nxt)
        truncated = ~terminated & (self.t >= MAX_STEPS)
        done = terminated | truncated
        final = nxt.copy()
        if done.any():
            nxt[done] = self.rng.uniform(-0.05, 0.05, size=(int(done.sum()), 4))
            self.t[done] = 0
        self.state = nxt
        return nxt.copy(), np.ones(self.n), terminated, truncated, final


def run_episode(policy, rng: np.random.Generator, record: bool = False):
    """Play one episode with ``policy(state) -> action``. Returns (return, states if record)."""
    env = CartPole(rng)
    s = env.reset()
    states, total = [s], 0.0
    while True:
        s, r, term, trunc = env.step(policy(s))
        total += r
        if record:
            states.append(s)
        if term or trunc:
            return total, (np.array(states) if record else None)
