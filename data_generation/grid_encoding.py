"""64x64 grid encoding of an airfoil (Phase 1, grid step).

Turns an airfoil outline into the fields the neural operator reads. The
contract (domain, conventions, channel order, scalings) is defined in
data_spec.md; the constants below are its single source in code.

Everything is expressed in chord units (chord = 1, leading edge at the origin,
trailing edge at x = 1), in the airfoil-fixed frame. Angle of attack is NOT
baked into the geometry, it enters as a constant channel.

Per-airfoil (alpha-independent) channels: sdf, mask, x, y.
Per-case scalar channels, constant over the grid: alpha, Re.

This script is deterministic (no randomness), so it has no --seed.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import shapely
from shapely.geometry import Polygon

# --- Grid contract (keep in sync with data_spec.md) ---------------------------
GRID_N = 64
X_MIN, X_MAX = -0.25, 1.25
Y_MIN, Y_MAX = -0.75, 0.75
DX = (X_MAX - X_MIN) / GRID_N
DY = (Y_MAX - Y_MIN) / GRID_N

# Scalar-channel scalings (model inputs): alpha in degrees / 10, log10(Re) / 6.
ALPHA_SCALE = 10.0
RE_LOG_SCALE = 6.0

CHANNELS = ["sdf", "mask", "x", "y", "alpha", "re"]


def cell_centres():
    """Cell-centre coordinates. Arrays are indexed [iy, ix]; iy increases with y
    (y up), ix increases with x."""
    xs = X_MIN + (np.arange(GRID_N) + 0.5) * DX
    ys = Y_MIN + (np.arange(GRID_N) + 0.5) * DY
    return np.meshgrid(xs, ys)  # (XX, YY), each (ny, nx)


def read_dat(path: str):
    """Read a Selig .dat file (name line, then x y rows). Returns (name, (n,2))."""
    with open(path, errors="replace") as f:
        name = f.readline().strip()
    coords = np.loadtxt(path, skiprows=1)
    return name, np.asarray(coords, dtype=float)


def make_polygon(coords: np.ndarray) -> Polygon:
    """Validated airfoil polygon. Raises ValueError for unusable outlines."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
        raise ValueError("need at least 3 (x, y) points")
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite coordinates")
    poly = Polygon(coords)
    if not poly.is_valid:
        raise ValueError("outline is not a valid simple polygon (self-intersecting?)")
    if poly.area <= 1e-9:
        raise ValueError("degenerate outline (zero area)")
    return poly


def _min_dist_to_edges(points: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to the closed outline."""
    a = coords
    b = np.roll(coords, -1, axis=0)  # closing edge included
    ab = b - a                                           # (M, 2)
    den = np.einsum("ij,ij->i", ab, ab)                  # (M,)
    ap = points[:, None, :] - a[None, :, :]              # (N, M, 2)
    t = np.einsum("nmj,mj->nm", ap, ab) / np.where(den > 0, den, 1.0)
    t = np.clip(np.where(den > 0, t, 0.0), 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return d.min(axis=1)


def signed_distance(points: np.ndarray, coords: np.ndarray, poly: Polygon | None = None):
    """Signed distance to the outline: negative inside, positive outside,
    in chord units. points is (N, 2)."""
    poly = make_polygon(coords) if poly is None else poly
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    d = _min_dist_to_edges(points, np.asarray(coords, dtype=float))
    inside = shapely.contains_xy(poly, points[:, 0], points[:, 1])
    return np.where(inside, -d, d)


def encode_airfoil(coords: np.ndarray):
    """Per-airfoil fields on the grid: sdf (float32), mask (uint8), shape (64, 64).
    mask is exactly (sdf < 0), so the two can never disagree."""
    poly = make_polygon(coords)
    xx, yy = cell_centres()
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    sdf = signed_distance(pts, coords, poly).reshape(GRID_N, GRID_N)
    return sdf.astype(np.float32), (sdf < 0).astype(np.uint8)


def sample_field(field: np.ndarray, x, y):
    """Bilinear sample of a (64, 64) [iy, ix] field at physical (x, y), border
    clamped. Matches torch grid_sample(align_corners=False, padding='border')."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fx = np.clip((x - X_MIN) / DX - 0.5, 0.0, GRID_N - 1.0)
    fy = np.clip((y - Y_MIN) / DY - 0.5, 0.0, GRID_N - 1.0)
    ix0 = np.minimum(np.floor(fx).astype(int), GRID_N - 2)
    iy0 = np.minimum(np.floor(fy).astype(int), GRID_N - 2)
    wx, wy = fx - ix0, fy - iy0
    f00 = field[iy0, ix0]
    f01 = field[iy0, ix0 + 1]
    f10 = field[iy0 + 1, ix0]
    f11 = field[iy0 + 1, ix0 + 1]
    return (f00 * (1 - wx) * (1 - wy) + f01 * wx * (1 - wy)
            + f10 * (1 - wx) * wy + f11 * wx * wy)


def xy_to_norm(x, y):
    """Physical (x, y) -> [-1, 1] coordinates for torch grid_sample
    (align_corners=False), where grid[..., 0] is x (width) and grid[..., 1] is y
    (height, increasing with iy)."""
    u = (np.asarray(x, dtype=float) - X_MIN) / (X_MAX - X_MIN) * 2.0 - 1.0
    v = (np.asarray(y, dtype=float) - Y_MIN) / (Y_MAX - Y_MIN) * 2.0 - 1.0
    return u, v


def alpha_channel_value(alpha_deg: float) -> float:
    return alpha_deg / ALPHA_SCALE


def re_channel_value(reynolds: float) -> float:
    return float(np.log10(reynolds) / RE_LOG_SCALE)


def assemble_input(sdf: np.ndarray, mask: np.ndarray, alpha_deg: float,
                   reynolds: float) -> np.ndarray:
    """The model input for one case: (6, 64, 64) float32 in CHANNELS order."""
    xx, yy = cell_centres()
    out = np.empty((len(CHANNELS), GRID_N, GRID_N), dtype=np.float32)
    out[0] = sdf
    out[1] = mask
    out[2] = xx
    out[3] = yy
    out[4] = alpha_channel_value(alpha_deg)
    out[5] = re_channel_value(reynolds)
    return out


def build_all(index_csv: str, airfoils_dir: str, out_path: str) -> None:
    with open(index_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    ids, sdfs, masks = [], [], []
    for r in rows:
        _, coords = read_dat(os.path.join(airfoils_dir, r["filename"]))
        sdf, mask = encode_airfoil(coords)
        ids.append(r["airfoil_id"])
        sdfs.append(sdf)
        masks.append(mask)
    xx, yy = cell_centres()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        airfoil_id=np.array(ids), sdf=np.stack(sdfs), mask=np.stack(masks),
        x=xx.astype(np.float32), y=yy.astype(np.float32),
        domain=np.array([X_MIN, X_MAX, Y_MIN, Y_MAX]),
    )
    print(f"Encoded {len(ids)} airfoils -> {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_csv", default=os.path.join(here, "airfoil_index.csv"))
    p.add_argument("--airfoils_dir", default=os.path.join(here, "airfoils"))
    p.add_argument("--out", default=os.path.join(here, "grids", "airfoil_grids.npz"))
    args = p.parse_args()
    build_all(args.index_csv, args.airfoils_dir, args.out)


if __name__ == "__main__":
    main()
