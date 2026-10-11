"""Visual check of the encoded grids: SDF field, zero level set and mask for six
airfoils spanning the dataset (thinnest, thickest, high camber, ...).
Writes grids/grid_check.png."""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid_encoding as ge  # noqa: E402

PICK = ["naca0006", "naca0018", "s1223", "goe387", "clarky", "rae2822"]


def main() -> int:
    g = np.load(os.path.join(HERE, "grids", "airfoil_grids.npz"), allow_pickle=False)
    ids = list(g["airfoil_id"])
    with open(os.path.join(HERE, "airfoil_index.csv"), newline="") as f:
        rows = {r["airfoil_id"]: r for r in csv.DictReader(f)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ext = [ge.X_MIN, ge.X_MAX, ge.Y_MIN, ge.Y_MAX]
    xx, yy = g["x"], g["y"]
    for ax, base in zip(axes.ravel(), PICK):
        aid = next(a for a in ids if rows[a]["baseline"] == base)
        i = ids.index(aid)
        sdf, mask = g["sdf"][i], g["mask"][i]
        im = ax.imshow(sdf, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-0.4, vmax=0.4, interpolation="nearest")
        ax.contour(xx, yy, sdf, levels=[0.0], colors="k", linewidths=1.4)
        ax.contour(xx, yy, sdf, levels=np.arange(-0.3, 0.76, 0.1), colors="0.5",
                   linewidths=0.4)
        _, co = ge.read_dat(os.path.join(HERE, "airfoils", rows[aid]["filename"]))
        ax.plot(co[:, 0], co[:, 1], "-", color="lime", lw=0.8)
        ax.contourf(xx, yy, mask, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.set_title(f"{aid} ({base}): {int(mask.sum())} mask cells", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(ge.X_MIN, ge.X_MAX)
        ax.set_ylim(ge.Y_MIN, ge.Y_MAX)
    fig.colorbar(im, ax=axes, shrink=0.7, label="signed distance (chords), negative inside")
    fig.suptitle("64x64 grid encoding: SDF, zero level set (black), true outline (green), mask (hatched)")
    out = os.path.join(HERE, "grids", "grid_check.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
