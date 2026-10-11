"""Dataset-wide checks on the encoded grids (run after grid_encoding.py).

Reports, over all airfoils: mask size, thin-section resolution, and how well
the 200 XFOIL surface nodes sit on the SDF zero level set (the property Head A
relies on when it samples the latent field at surface points).
Writes grids/grid_stats.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid_encoding as ge  # noqa: E402


def main() -> int:
    g = np.load(os.path.join(HERE, "grids", "airfoil_grids.npz"), allow_pickle=False)
    ids, sdf, mask = g["airfoil_id"], g["sdf"], g["mask"]
    lines = [f"airfoils: {len(ids)}   grid {sdf.shape[1]}x{sdf.shape[2]}   "
             f"domain x[{g['domain'][0]}, {g['domain'][1]}] y[{g['domain'][2]}, {g['domain'][3]}]"]

    n_cells = mask.reshape(len(ids), -1).sum(axis=1)
    lines.append(f"mask cells per airfoil: min {n_cells.min()}  median {int(np.median(n_cells))}  max {n_cells.max()}")
    lines.append(f"airfoils with an empty mask: {int((n_cells == 0).sum())}")

    # thickness in cells at the thickest column (rows of inside cells in a column)
    col_max = mask.sum(axis=1).max(axis=1)  # (n_airfoils,)
    lines.append(f"max cells across the wing thickness: min {col_max.min()}  median {int(np.median(col_max))}  max {col_max.max()}")
    lines.append(f"airfoils with a thickest column of <= 2 cells: {int((col_max <= 2).sum())}")

    # surface nodes on the zero level set
    dist_cells, n_done = [], 0
    for i, aid in enumerate(ids):
        p = os.path.join(HERE, "xfoil_data", "cp", f"{aid}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=False)
        s = np.abs(ge.sample_field(sdf[i], d["x"], d["y"])) / ge.DX
        dist_cells.append(s)
        n_done += 1
    allc = np.concatenate(dist_cells)
    per_af_max = np.array([s.max() for s in dist_cells])
    lines.append(f"XFOIL surface nodes checked: {n_done} airfoils x 200 nodes")
    lines.append(f"|SDF| at surface nodes (cells): mean {allc.mean():.3f}  p99 {np.percentile(allc, 99):.3f}  max {allc.max():.3f}")
    lines.append(f"airfoils whose worst node is > 0.75 cell off the surface: {int((per_af_max > 0.75).sum())}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "grids", "grid_stats.txt"), "w") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
