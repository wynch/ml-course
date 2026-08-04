"""The quadratic feature map, written out longhand.

The kernel trick is usually sold as "you never have to compute φ". That is
true and it is also why it feels like a magic trick. For the homogeneous
quadratic kernel in 2D the map is small enough to print, so here it is:

    φ(x₁, x₂) = (x₁², √2·x₁x₂, x₂²)

and a two-line expansion shows it reproduces the kernel exactly:

    ⟨φ(a), φ(b)⟩ = a₁²b₁² + 2a₁a₂b₁b₂ + a₂²b₂²
                 = (a₁b₁ + a₂b₂)²
                 = (aᵀb)²  =  K(a, b).

The √2 is not decoration — drop it and the cross term is counted once instead
of twice and the identity breaks. :func:`check_identity` asserts it.

Two different 3D pictures get used for "lifting circles until a plane can cut
them" and they are not the same map:

- :func:`phi_quadratic` is the *honest* one: the actual feature space of the
  quadratic kernel. Concentric rings land on a cone, and the plane
  z₁ + z₃ = r² separates them because z₁ + z₃ = x₁² + x₂² = ‖x‖².
- :func:`phi_paraboloid` is the *intuitive* one, (x₁, x₂, ‖x‖²), which keeps
  the original coordinates and adds a height. It is easier to look at, it is
  also a legitimate feature map (its kernel is aᵀb + ‖a‖²‖b‖²), but it is not
  the map hiding inside (aᵀb)².
"""

from __future__ import annotations

import numpy as np

SQRT2 = float(np.sqrt(2.0))


def phi_quadratic(X: np.ndarray) -> np.ndarray:
    """φ(x) = (x₁², √2·x₁x₂, x₂²) — the feature map of K(a, b) = (aᵀb)²."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack([X[:, 0] ** 2, SQRT2 * X[:, 0] * X[:, 1], X[:, 1] ** 2])


def phi_paraboloid(X: np.ndarray) -> np.ndarray:
    """φ(x) = (x₁, x₂, x₁² + x₂²) — the picture-book lift onto a paraboloid."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack([X[:, 0], X[:, 1], (X**2).sum(1)])


def check_identity(X: np.ndarray) -> float:
    """Max |⟨φ(a), φ(b)⟩ − (aᵀb)²| over all pairs. Should be ~1e-15."""
    F = phi_quadratic(X)
    return float(np.max(np.abs(F @ F.T - (X @ X.T) ** 2)))


def quadratic_plane(w: np.ndarray, b: float, extent: float = 5.0, n: int = 24):
    """Sample the plane wᵀz + b = 0 over a grid, for plotting in feature space.

    Returns ``(Z1, Z2, Z3)`` meshes with z₃ solved from the plane equation.
    Requires ``w[2] != 0``.
    """
    g = np.linspace(0.0, extent, n)
    Z1, Z2g = np.meshgrid(g, np.linspace(-extent, extent, n))
    Z3 = -(w[0] * Z1 + w[1] * Z2g + b) / w[2]
    return Z1, Z2g, Z3
