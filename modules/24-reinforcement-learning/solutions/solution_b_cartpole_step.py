"""Reference solution, exercise (b): one step of CartPole from its equations of motion."""

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
    sin, cos = math.sin(theta), math.cos(theta)
    temp = (force + MASS_POLE * HALF_LENGTH * theta_dot**2 * sin) / total
    theta_dd = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * cos**2 / total))
    x_dd = temp - MASS_POLE * HALF_LENGTH * theta_dd * cos / total
    return np.array([x + TAU * x_dot, x_dot + TAU * x_dd, theta + TAU * theta_dot, theta_dot + TAU * theta_dd])


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
