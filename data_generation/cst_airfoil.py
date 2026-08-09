"""CST (Class-Shape Transformation) airfoil parameterization.

Implements the Kulfan CST method: an airfoil surface is the product of a
class function (a generic round-nose/sharp-tail template) and a shape
function (a Bernstein-polynomial perturbation), plus a linear
trailing-edge thickness term. See data_spec.md at the repo root for the
output format contract.
"""

from __future__ import annotations

from math import comb

import numpy as np

N_COEFFS = 5  # CST coefficients per surface (upper and lower each have 5)


def cosine_spacing(n_points: int) -> np.ndarray:
    """Cosine-spaced x-coordinates in [0, 1], bunched near x=0 and x=1."""
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def class_function(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """Standard subsonic-airfoil class function C(x) = x^n1 * (1-x)^n2."""
    return np.power(x, n1) * np.power(1.0 - x, n2)


def bernstein_basis(x: np.ndarray, i: int, n: int) -> np.ndarray:
    """i-th Bernstein basis polynomial of degree n, evaluated at x."""
    return comb(n, i) * np.power(x, i) * np.power(1.0 - x, n - i)


def shape_function(x: np.ndarray, coeffs) -> np.ndarray:
    """Bernstein-polynomial shape function S(x) = sum_i A_i * K_i,n(x).

    `coeffs` has N_COEFFS entries -> degree n = N_COEFFS - 1 Bernstein basis.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs) - 1
    return sum(coeffs[i] * bernstein_basis(x, i, n) for i in range(n + 1))


def cst_surface(x: np.ndarray, coeffs, te_half_thickness: float,
                 n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """CST surface: y(x) = C(x) * S(x) + x * te_half_thickness."""
    return class_function(x, n1, n2) * shape_function(x, coeffs) + x * te_half_thickness


def generate_airfoil_coordinates(upper_coeffs, lower_coeffs, n_points: int = 100,
                                  te_thickness: float = 0.002):
    """Generate one airfoil's surface coordinates.

    Returns (x, y_upper, y_lower, coords):
      - x, y_upper, y_lower: arrays on the same cosine-spaced x grid
      - coords: combined (2*n_points-1, 2) array in Selig format
        (upper TE -> LE -> lower TE), ready to write as a .dat file
    """
    x = cosine_spacing(n_points)
    y_upper = cst_surface(x, upper_coeffs, te_thickness / 2.0)
    y_lower = cst_surface(x, lower_coeffs, -te_thickness / 2.0)

    upper_pts = np.column_stack([x[::-1], y_upper[::-1]])  # TE -> LE
    lower_pts = np.column_stack([x[1:], y_lower[1:]])      # LE (excl, shared) -> TE
    coords = np.vstack([upper_pts, lower_pts])

    return x, y_upper, y_lower, coords


def coordinates_from_cst(upper_coeffs, lower_coeffs, te_upper: float,
                          te_lower: float, n_points: int = 100):
    """Like generate_airfoil_coordinates, but with an independent trailing-edge
    offset per surface.

    Real airfoils fitted from coordinate files rarely have a perfectly
    symmetric TE gap, so upper and lower each carry their own offset
    instead of sharing +/- half of one thickness.

    Returns (x, y_upper, y_lower, coords) exactly as
    generate_airfoil_coordinates does.
    """
    x = cosine_spacing(n_points)
    y_upper = cst_surface(x, upper_coeffs, te_upper)
    y_lower = cst_surface(x, lower_coeffs, te_lower)

    upper_pts = np.column_stack([x[::-1], y_upper[::-1]])  # TE -> LE
    lower_pts = np.column_stack([x[1:], y_lower[1:]])      # LE (excl, shared) -> TE
    coords = np.vstack([upper_pts, lower_pts])

    return x, y_upper, y_lower, coords


def thickness_and_camber(x: np.ndarray, y_upper: np.ndarray, y_lower: np.ndarray):
    """Max thickness and signed max camber, both as chord fractions.

    max_camber is the camber-line value at the point of maximum |camber|,
    so its sign reflects whether the airfoil is cambered up or down there.
    """
    thickness_dist = y_upper - y_lower
    camber_line = 0.5 * (y_upper + y_lower)
    max_thickness = float(np.max(thickness_dist))
    max_camber = float(camber_line[np.argmax(np.abs(camber_line))])
    return max_thickness, max_camber


def is_valid_airfoil(x: np.ndarray, y_upper: np.ndarray, y_lower: np.ndarray,
                      min_thickness: float = 0.02, max_thickness_limit: float = 0.22) -> bool:
    """Geometry validity filter.

    Rejects self-intersecting shapes (upper surface below lower surface
    anywhere) and shapes thinner than `min_thickness` or thicker than
    `max_thickness_limit` (both as chord fractions).
    """
    thickness_dist = y_upper - y_lower
    if np.any(thickness_dist < -1e-9):
        return False
    max_t = float(np.max(thickness_dist))
    if max_t < min_thickness or max_t > max_thickness_limit:
        return False
    return True


def write_dat_file(path: str, coords: np.ndarray, name: str) -> None:
    """Write coordinates to a Selig-format .dat file (name header + x y rows)."""
    with open(path, "w") as f:
        f.write(f"{name}\n")
        for xi, yi in coords:
            f.write(f"{xi: .6f} {yi: .6f}\n")
