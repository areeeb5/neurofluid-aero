"""Unit tests for grid_encoding.py. Run: python data_generation/test_grid_encoding.py

Known-good cases (exact circle, symmetric airfoil, real XFOIL surface nodes)
and known-bad cases (self-intersecting and degenerate outlines) per the repo
convention that every filter/validity check is tested before it touches data.
Exits nonzero on any failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import shapely
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grid_encoding as ge  # noqa: E402
from cst_airfoil import generate_airfoil_coordinates  # noqa: E402

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def circle(cx: float, cy: float, r: float, n: int = 2000) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def sym_airfoil(scale: float = 1.0) -> np.ndarray:
    up = np.array([0.17, 0.16, 0.15, 0.14, 0.13]) * scale
    return generate_airfoil_coordinates(up, -up, n_points=100, te_thickness=0.002)[3]


def cambered_airfoil(scale: float) -> np.ndarray:
    up = np.array([0.20, 0.19, 0.17, 0.15, 0.12]) * scale
    lo = -np.array([0.10, 0.08, 0.07, 0.06, 0.05]) * scale
    return generate_airfoil_coordinates(up, lo, n_points=100, te_thickness=0.002)[3]


def main() -> int:
    print(f"Grid: {ge.GRID_N}x{ge.GRID_N}, x [{ge.X_MIN}, {ge.X_MAX}], "
          f"y [{ge.Y_MIN}, {ge.Y_MAX}], cell {ge.DX:.5f} x {ge.DY:.5f} chords\n")
    xx, yy = ge.cell_centres()
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    # 1. Circle: SDF must equal the analytic r - R.
    R = 0.2
    c = circle(0.5, 0.0, R)
    sdf = ge.signed_distance(pts, c).reshape(ge.GRID_N, ge.GRID_N)
    exact = np.hypot(xx - 0.5, yy - 0.0) - R
    err = float(np.max(np.abs(sdf - exact)))
    check("circle SDF matches analytic r - R", err < 1e-4, f"max err {err:.2e}")

    # 2. Circle: mask area close to pi R^2.
    sdf_c, mask_c = ge.encode_airfoil(c)
    area = mask_c.sum() * ge.DX * ge.DY
    rel = abs(area - np.pi * R**2) / (np.pi * R**2)
    check("circle mask area within 3% of pi R^2", rel < 0.03, f"{rel*100:.2f}%")

    # 3. Symmetric airfoil: SDF and mask symmetric about y = 0.
    sym = sym_airfoil()
    sdf_s, mask_s = ge.encode_airfoil(sym)
    asym = float(np.max(np.abs(sdf_s - sdf_s[::-1, :])))
    check("symmetric airfoil: SDF symmetric about y=0", asym < 1e-6, f"max asym {asym:.2e}")
    check("symmetric airfoil: mask symmetric about y=0",
          bool(np.array_equal(mask_s, mask_s[::-1, :])))

    # 4. SDF is zero ON the outline (vertices and edge midpoints).
    mids = 0.5 * (sym + np.roll(sym, -1, axis=0))
    on = np.vstack([sym, mids])
    d_on = np.abs(ge.signed_distance(on, sym))
    check("SDF = 0 on outline vertices and edge midpoints", float(d_on.max()) < 1e-9,
          f"max {d_on.max():.2e}")

    # 5. Sign: positive outside, negative inside.
    far = ge.signed_distance(np.array([[-0.2, -0.7]]), sym)[0]
    inside = ge.signed_distance(np.array([[0.30, 0.0]]), sym)[0]
    check("sign: positive outside, negative inside", far > 0 and inside < 0,
          f"far {far:+.3f}, inside {inside:+.3f}")

    # 6. Area: mask area within 3% of the true polygon area (3 airfoils).
    for label, co in (("12% symmetric", sym), ("thick 18%", sym_airfoil(1.5)),
                      ("cambered", cambered_airfoil(1.0))):
        _, m = ge.encode_airfoil(co)
        true_area = Polygon(co).area
        rel = abs(m.sum() * ge.DX * ge.DY - true_area) / true_area
        check(f"mask area within 3% of polygon area ({label})", rel < 0.03,
              f"{rel*100:.2f}%")

    # 7. Eikonal property: |grad SDF| = 1 away from the surface and the wake line.
    gy, gx = np.gradient(sdf_s, ge.DY, ge.DX)
    gmag = np.hypot(gx, gy)
    sel = (sdf_s > 3 * ge.DX) & ~((xx > 0.9) & (np.abs(yy) < 3 * ge.DY))
    dev = np.abs(gmag[sel] - 1.0)
    check("|grad SDF| = 1 away from surface (99th pct dev < 0.05)",
          float(np.percentile(dev, 99)) < 0.05,
          f"99th pct {np.percentile(dev, 99):.3f}, max {dev.max():.3f}, n={sel.sum()}")

    # 8. Bad outlines are rejected.
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    for label, bad in (("self-intersecting bow-tie", bowtie),
                       ("two points", np.array([[0, 0], [1, 0]], dtype=float)),
                       ("zero-area (collinear)", np.array([[0, 0], [0.5, 0], [1, 0]], dtype=float)),
                       ("NaN coordinate", np.array([[0, 0], [1, np.nan], [1, 0]], dtype=float))):
        try:
            ge.make_polygon(bad)
            check(f"rejects {label}", False, "was accepted")
        except ValueError:
            check(f"rejects {label}", True)

    # 9. Real data: XFOIL surface nodes lie on the SDF zero level set.
    cp_path = os.path.join(HERE, "xfoil_data", "cp", "airfoil_0043.npz")
    dat_path = os.path.join(HERE, "airfoils", "airfoil_0043.dat")
    if os.path.exists(cp_path) and os.path.exists(dat_path):
        _, coords = ge.read_dat(dat_path)
        sdf_r, _ = ge.encode_airfoil(coords)
        d = np.load(cp_path, allow_pickle=False)
        s = np.abs(ge.sample_field(sdf_r, d["x"], d["y"]))
        check("real airfoil: |SDF| at 200 XFOIL nodes, mean < 0.25 cell", float(s.mean()) < 0.25 * ge.DX,
              f"mean {s.mean()/ge.DX:.3f} cell, max {s.max()/ge.DX:.3f} cell")
        check("real airfoil: |SDF| at 200 XFOIL nodes, max < 0.75 cell", float(s.max()) < 0.75 * ge.DX)
    else:
        print("  [SKIP] real-airfoil surface-node test (data files not found)")

    # 10. Input assembly: shape, order, scalings.
    inp = ge.assemble_input(sdf_s, mask_s, 4.0, 1e6)
    ok = (inp.shape == (6, 64, 64) and np.allclose(inp[4], 0.4)
          and np.allclose(inp[5], 1.0) and np.allclose(inp[2], xx) and np.allclose(inp[3], yy))
    check("assemble_input: (6,64,64), alpha=4 deg -> 0.4, Re=1e6 -> 1.0, x/y channels", ok)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("All grid-encoding tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
