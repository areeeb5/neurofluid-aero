"""Fit CST coefficients to real airfoil coordinate files.

With the class function and Bernstein basis fixed, a CST surface is
LINEAR in its coefficients:

    y(x) = C(x) * sum_i A_i * K_i(x)  +  x * te_offset

so fitting is ordinary linear least-squares on A, solved with
numpy.linalg.lstsq - no nonlinear optimizer required.

The design matrix is built as C(x_j) * K_i(x_j) rather than dividing the
data through by C(x); C(x) vanishes at both x=0 and x=1, so dividing
would blow up exactly at the leading and trailing edge.
"""

from __future__ import annotations

import os

import numpy as np

from cst_airfoil import N_COEFFS, bernstein_basis, class_function, cst_surface


# ---------------------------------------------------------------------------
# Reading UIUC coordinate files (two formats in the wild)
# ---------------------------------------------------------------------------

def _numeric_pairs(lines):
    pairs = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pairs.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return pairs


def normalize_surfaces(xu, yu, xl, yl):
    """Put an airfoil in canonical position: LE at (0,0), TE midpoint at (1,0).

    UIUC files are not consistently normalized - e.g. sd7062's leading
    edge sits at (0.00027, -0.00327) and e387's at (0.00044, +0.00234).
    That matters because the CST class function pins y(0) = 0, so fitting
    an un-normalized airfoil forces an irreducible error at the nose and
    can distort the fit enough to make the upper surface cross below the
    lower one near the leading edge.

    Translates the LE to the origin, rotates the chord line horizontal,
    and scales the chord to 1.
    """
    le = np.array([0.5 * (xu[0] + xl[0]), 0.5 * (yu[0] + yl[0])])
    te = np.array([0.5 * (xu[-1] + xl[-1]), 0.5 * (yu[-1] + yl[-1])])

    chord_vec = te - le
    chord = float(np.hypot(*chord_vec))
    if chord <= 0:
        raise ValueError("degenerate chord")
    theta = float(np.arctan2(chord_vec[1], chord_vec[0]))
    c, s = np.cos(-theta), np.sin(-theta)

    def xf(x, y):
        dx, dy = np.asarray(x) - le[0], np.asarray(y) - le[1]
        xr = (dx * c - dy * s) / chord
        yr = (dx * s + dy * c) / chord
        return np.clip(xr, 0.0, 1.0), yr

    xu2, yu2 = xf(xu, yu)
    xl2, yl2 = xf(xl, yl)
    return xu2, yu2, xl2, yl2


def read_airfoil_dat(path: str, normalize: bool = True):
    """Read a UIUC .dat file in either Selig or Lednicer format.

    Returns (name, xu, yu, xl, yl) with each surface ordered leading edge
    -> trailing edge, normalized to LE (0,0) / TE (1,0) unless disabled.

    - Selig: one contiguous loop, upper TE -> LE -> lower TE.
    - Lednicer: a counts line (e.g. "66. 66."), then the upper surface
      LE -> TE, then the lower surface LE -> TE.
    """
    raw = open(path, errors="replace").read().splitlines()
    name = raw[0].strip() if raw else os.path.basename(path)

    pairs = _numeric_pairs(raw[1:])
    if not pairs:
        raise ValueError(f"{path}: no coordinate data")

    first = pairs[0]
    # A Lednicer file leads with point COUNTS, which are >= 2 and never
    # valid chord-normalized coordinates.
    if first[0] >= 2.0 and first[1] >= 2.0:
        n_u, n_l = int(round(first[0])), int(round(first[1]))
        body = pairs[1:]
        if len(body) < n_u + n_l:
            raise ValueError(f"{path}: Lednicer counts {n_u}+{n_l} exceed "
                             f"{len(body)} points")
        up = np.array(body[:n_u], dtype=float)
        lo = np.array(body[n_u:n_u + n_l], dtype=float)
        xu, yu, xl, yl = up[:, 0], up[:, 1], lo[:, 0], lo[:, 1]
    else:
        pts = np.array(pairs, dtype=float)
        le = int(np.argmin(pts[:, 0]))
        upper = pts[:le + 1][::-1]   # TE -> LE  =>  LE -> TE
        lower = pts[le:]             # LE -> TE
        xu, yu, xl, yl = upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]

    # Normalize ordering LE -> TE and guard against duplicate x.
    if xu[0] > xu[-1]:
        xu, yu = xu[::-1], yu[::-1]
    if xl[0] > xl[-1]:
        xl, yl = xl[::-1], yl[::-1]

    if normalize:
        xu, yu, xl, yl = normalize_surfaces(xu, yu, xl, yl)
    return name, xu, yu, xl, yl


# ---------------------------------------------------------------------------
# Least-squares CST fit
# ---------------------------------------------------------------------------

def fit_surface_cst(x: np.ndarray, y: np.ndarray, n_coeffs: int = N_COEFFS):
    """Least-squares fit of one surface. Returns (coeffs, te_offset, max_err)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # TE offset is pinned to the surface's own y at x=1 so the fitted
    # curve reproduces the real trailing edge exactly.
    te_offset = float(np.interp(1.0, x, y))

    n = n_coeffs - 1
    C = class_function(x)
    M = np.column_stack([C * bernstein_basis(x, i, n) for i in range(n_coeffs)])
    b = y - x * te_offset

    coeffs, *_ = np.linalg.lstsq(M, b, rcond=None)
    max_err = float(np.max(np.abs(cst_surface(x, coeffs, te_offset) - y)))
    return coeffs, te_offset, max_err


def fit_airfoil_cst(path: str, n_coeffs: int = N_COEFFS) -> dict:
    """Fit both surfaces of a real airfoil file."""
    name, xu, yu, xl, yl = read_airfoil_dat(path)
    cu, teu, eu = fit_surface_cst(xu, yu, n_coeffs)
    cl, tel, el = fit_surface_cst(xl, yl, n_coeffs)
    return {
        "name": name, "path": path,
        "upper_coeffs": cu, "lower_coeffs": cl,
        "te_upper": teu, "te_lower": tel,
        "max_err_upper": eu, "max_err_lower": el,
        "max_err": max(eu, el),
        "xu": xu, "yu": yu, "xl": xl, "yl": yl,
    }
