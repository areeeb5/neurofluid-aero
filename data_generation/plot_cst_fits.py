"""Validation plots for the CST fits against the real UIUC baselines.

Produces:
  plots/cst_fit_validation.png  - all baselines, fitted curve over real points
  plots/cst_fit_detail.png      - 3 baselines up close, with residuals
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cst_airfoil import cst_surface
from cst_fit import fit_airfoil_cst
from download_baselines import UIUC_FILENAME

LE_EXCLUDE = 0.01  # CST pins y(0)=0; report error away from that singular point


def fit_all(baselines_dir: str, n_coeffs: int):
    out = {}
    for nm in UIUC_FILENAME:
        out[nm] = fit_airfoil_cst(os.path.join(baselines_dir, nm + ".dat"), n_coeffs)
    return out


def err_excl_le(f):
    e = 0.0
    for x, y, c, te in ((f["xu"], f["yu"], f["upper_coeffs"], f["te_upper"]),
                        (f["xl"], f["yl"], f["lower_coeffs"], f["te_lower"])):
        m = x >= LE_EXCLUDE
        if m.any():
            e = max(e, float(np.max(np.abs(cst_surface(x[m], c, te) - y[m]))))
    return e


def plot_overview(fits, out_path: str, n_coeffs: int):
    n = len(fits)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).flatten()

    xf = np.linspace(0, 1, 400)
    for ax, (nm, f) in zip(axes, fits.items()):
        ax.plot(f["xu"], f["yu"], "o", ms=2.4, color="0.35", label="UIUC data")
        ax.plot(f["xl"], f["yl"], "o", ms=2.4, color="0.35")
        ax.plot(xf, cst_surface(xf, f["upper_coeffs"], f["te_upper"]),
                "-", lw=1.3, color="tab:red", label="CST fit")
        ax.plot(xf, cst_surface(xf, f["lower_coeffs"], f["te_lower"]),
                "-", lw=1.3, color="tab:red")
        ax.set_title(f"{nm}   max err {err_excl_le(f):.5f}c (x>{LE_EXCLUDE:g})",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[len(fits):]:
        ax.axis("off")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"CST fit validation - {n_coeffs} coefficients per surface, "
                 f"linear least-squares", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_detail(fits, names, out_path: str, n_coeffs: int):
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 6),
                             gridspec_kw={"height_ratios": [2, 1]})
    xf = np.linspace(0, 1, 600)
    for j, nm in enumerate(names):
        f = fits[nm]
        ax = axes[0, j]
        ax.plot(f["xu"], f["yu"], "o", ms=4, color="0.3", label="UIUC data")
        ax.plot(f["xl"], f["yl"], "o", ms=4, color="0.3")
        ax.plot(xf, cst_surface(xf, f["upper_coeffs"], f["te_upper"]),
                "-", lw=1.5, color="tab:red", label="CST fit")
        ax.plot(xf, cst_surface(xf, f["lower_coeffs"], f["te_lower"]),
                "-", lw=1.5, color="tab:red")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{nm}", fontsize=11)
        if j == 0:
            ax.legend(fontsize=8)

        axr = axes[1, j]
        for x, y, c, te, lab in ((f["xu"], f["yu"], f["upper_coeffs"], f["te_upper"], "upper"),
                                 (f["xl"], f["yl"], f["lower_coeffs"], f["te_lower"], "lower")):
            axr.plot(x, (cst_surface(x, c, te) - y) * 1000, ".-", ms=4, lw=1, label=lab)
        axr.axhline(0, color="k", lw=0.6)
        axr.axhspan(-1, 1, color="tab:green", alpha=0.12)
        axr.axvspan(0, LE_EXCLUDE, color="tab:orange", alpha=0.15)
        axr.set_xlabel("x/c"); axr.set_ylabel("residual (1e-3 c)")
        axr.grid(alpha=0.3); axr.legend(fontsize=8)
        axr.set_title("green band = +/-0.001c target;  orange = LE region",
                      fontsize=8)
    fig.suptitle(f"CST fit detail - {n_coeffs} coefficients per surface", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines_dir", default=os.path.join(here, "baselines"))
    p.add_argument("--plots_dir", default=os.path.join(here, "plots"))
    p.add_argument("--n_coeffs", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    fits = fit_all(args.baselines_dir, args.n_coeffs)

    print(f"CST fit quality ({args.n_coeffs} coefficients/surface)")
    print(f"{'baseline':14s}{'max err (all)':>15s}{'max err (x>0.01)':>18s}")
    for nm, f in fits.items():
        print(f"{nm:14s}{f['max_err']:15.6f}{err_excl_le(f):18.6f}")

    plot_overview(fits, os.path.join(args.plots_dir, "cst_fit_validation.png"),
                  args.n_coeffs)
    plot_detail(fits, ["naca4412", "clarky", "s1223"],
                os.path.join(args.plots_dir, "cst_fit_detail.png"), args.n_coeffs)


if __name__ == "__main__":
    main()
