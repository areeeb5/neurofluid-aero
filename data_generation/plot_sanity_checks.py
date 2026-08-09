"""Sanity-check plots for the generated airfoil dataset.

Reads airfoil_index.csv + coordinate files under airfoils/ (produced by
generate_airfoils.py) and produces:
  1. plots/sample_grid.png              - random airfoil outlines, colored by baseline
  2. plots/thickness_camber_scatter.png - design-space coverage, clustered by baseline
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p", "d", "8", "H"]


def load_index(index_csv: str):
    with open(index_csv, newline="") as f:
        return list(csv.DictReader(f))


def load_coords(airfoils_dir: str, filename: str) -> np.ndarray:
    return np.loadtxt(os.path.join(airfoils_dir, filename), skiprows=1)


def baseline_style(records):
    """Stable color+marker per baseline."""
    names = sorted({r["baseline"] for r in records})
    cmap = plt.get_cmap("tab20")
    return {nm: (cmap(i % 20), MARKERS[i % len(MARKERS)])
            for i, nm in enumerate(names)}, names


def plot_sample_grid(records, airfoils_dir, out_path, style,
                     n_samples=12, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=n_samples, replace=False)
    sample = [records[i] for i in idx]

    ncols = 4
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, rec in zip(axes, sample):
        coords = load_coords(airfoils_dir, rec["filename"])
        color = style[rec["baseline"]][0]
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=1.3)
        ax.set_title(f"{rec['airfoil_id']}\nfrom {rec['baseline']}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle("Random sample of generated airfoils (colored by real baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample grid -> {out_path}")


def plot_thickness_camber(records, out_path, style, names):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for nm in names:
        sub = [r for r in records if r["baseline"] == nm]
        t = np.array([float(r["max_thickness"]) for r in sub]) * 100
        c = np.array([float(r["max_camber"]) for r in sub]) * 100
        color, marker = style[nm]
        ax.scatter(t, c, s=26, alpha=0.75, color=color, marker=marker,
                   edgecolors="none", label=f"{nm} (n={len(sub)})")

    ax.axvline(2, color="gray", ls="--", lw=0.9)
    ax.axvline(22, color="gray", ls="--", lw=0.9)
    ax.text(2.3, ax.get_ylim()[1] * 0.95, "filter min 2%", fontsize=7, color="gray")
    ax.text(21.7, ax.get_ylim()[1] * 0.95, "filter max 22%", fontsize=7,
            color="gray", ha="right")
    ax.set_xlabel("Max thickness (% chord)")
    ax.set_ylabel("Max camber (% chord)")
    ax.set_title(f"Design-space coverage, clustered by real baseline "
                 f"(n={len(records)})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved thickness/camber scatter -> {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_csv", default=os.path.join(here, "airfoil_index.csv"))
    p.add_argument("--airfoils_dir", default=os.path.join(here, "airfoils"))
    p.add_argument("--plots_dir", default=os.path.join(here, "plots"))
    p.add_argument("--n_samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    records = load_index(args.index_csv)
    style, names = baseline_style(records)

    plot_sample_grid(records, args.airfoils_dir,
                     os.path.join(args.plots_dir, "sample_grid.png"),
                     style, n_samples=args.n_samples, seed=args.seed)
    plot_thickness_camber(records,
                          os.path.join(args.plots_dir, "thickness_camber_scatter.png"),
                          style, names)


if __name__ == "__main__":
    main()
