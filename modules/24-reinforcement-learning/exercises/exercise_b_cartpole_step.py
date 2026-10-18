"""Exercise (b): one step of CartPole from its equations of motion.

The state is (x, x_dot, theta, theta_dot): cart position and speed, pole
angle from upright and its rate. Action 1 pushes the cart with +10 N, action
0 with -10 N. With M the cart's mass, m the pole's, l half the pole's length:

    temp     = (F + m l theta_dot^2 sin theta) / (M + m)
    theta_dd = (g sin theta - cos theta * temp) / (l (4/3 - m cos^2 theta / (M + m)))
    x_dd     = temp - m l theta_dd cos theta / (M + m)

then one Euler step of tau: positions move with the old speeds, speeds with
the new accelerations.

Predict before you run: from the state (0, 0, 0, 0), push right once. Which
of the four numbers change, and which way does the pole start to turn? The
test checks your step against exact fractions worked out by hand.

    uv run python ../exercises/exercise_b_cartpole_step.py
    uv run pytest tests/test_solutions.py -k cartpole -v
"""

from __future__ import annotations

import math

import numpy as np

GRAVITY, MASS_CART, MASS_POLE, HALF_LENGTH = 9.8, 1.0, 0.1, 0.5
FORCE_MAG, TAU = 10.0, 0.02


def step(state, action):
    """One Euler step; ``state`` is a length-4 array, ``action`` 0 or 1. Returns the new state."""
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    total = MASS_CART + MASS_POLE
    # TODO(you): temp, theta_dd and x_dd from the three equations above.
    # TODO(you): return np.array([x + TAU x_dot, x_dot + TAU x_dd, theta + TAU theta_dot, theta_dot + TAU theta_dd]).
    ...


def main() -> None:
    s = np.zeros(4)
    print("push right from rest:", step(s, 1))
    print("push left from rest: ", step(s, 0))
    s = np.array([0.0, 0.0, 0.05, 0.0])
    for t in range(10):
        s = step(s, 1)
    print("pole tilted 0.05 rad, ten pushes right:", np.round(s, 4))


if __name__ == "__main__":
    main()
