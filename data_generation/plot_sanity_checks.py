"""Sanity-check plots for the generated CST airfoil dataset.

Reads airfoil_index.csv + coordinate files under airfoils/ (produced by
generate_airfoils.py) and produces:
  1. plots/sample_grid.png -- a grid of random airfoil outlines
  2. plots/thickness_camber_scatter.png -- design-space coverage
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_index(index_csv: str):
    with open(index_csv, newline="") as f:
        return list(csv.DictReader(f))


def load_coords(airfoils_dir: str, filename: str) -> np.ndarray:
    return np.loadtxt(os.path.join(airfoils_dir, filename), skiprows=1)


def plot_sample_grid(records, airfoils_dir: str, out_path: str,
                      n_samples: int = 12, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=n_samples, replace=False)
    sample = [records[i] for i in idx]

    ncols = 4
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, rec in zip(axes, sample):
        coords = load_coords(airfoils_dir, rec["filename"])
        ax.plot(coords[:, 0], coords[:, 1], color="tab:blue", lw=1.2)
        ax.set_title(rec["airfoil_id"], fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle("Random sample of generated CST airfoils")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample grid -> {out_path}")


def plot_thickness_camber(records, out_path: str) -> None:
    thickness = np.array([float(r["max_thickness"]) for r in records]) * 100
    camber = np.array([float(r["max_camber"]) for r in records]) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(thickness, camber, s=14, alpha=0.6, color="tab:orange", edgecolors="none")
    ax.axvline(2, color="gray", ls="--", lw=0.8)
    ax.axvline(22, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Max thickness (% chord)")
    ax.set_ylabel("Max camber (% chord)")
    ax.set_title(f"Design-space coverage (n={len(records)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved thickness/camber scatter -> {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_csv", type=str, default=os.path.join(here, "airfoil_index.csv"))
    parser.add_argument("--airfoils_dir", type=str, default=os.path.join(here, "airfoils"))
    parser.add_argument("--plots_dir", type=str, default=os.path.join(here, "plots"))
    parser.add_argument("--n_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    records = load_index(args.index_csv)

    plot_sample_grid(records, args.airfoils_dir,
                      os.path.join(args.plots_dir, "sample_grid.png"),
                      n_samples=args.n_samples, seed=args.seed)
    plot_thickness_camber(records, os.path.join(args.plots_dir, "thickness_camber_scatter.png"))


if __name__ == "__main__":
    main()
