"""Validation checks for the XFOIL installation, using NACA 0012.

NACA 0012 is symmetric, which gives us exact, known-a-priori expectations
that any correct setup must reproduce:
  1. CL(alpha=0) == 0
  2. The Cp distribution at alpha=0 is identical on upper and lower surfaces
  3. The polar is antisymmetric: CL(-a) == -CL(a), CD(-a) == CD(a), and the
     transition locations mirror: Top_Xtr(-a) == Bot_Xtr(a)

If any of these fail, something is wrong with the XFOIL setup and no batch
run (Step 4) should be trusted. Run after generating the files in
xfoil_validation/ (see data_spec.md).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Tolerances. XFOIL's viscous solve is iterative, so "symmetric" means
# symmetric to solver tolerance, not to machine precision.
TOL_CL_ZERO = 1e-4      # |CL| at alpha=0
TOL_CP_SYM = 5e-3       # max |Cp_upper - Cp_lower| at alpha=0
TOL_POLAR_ANTISYM = 5e-4  # |CL(-a) + CL(a)| and |CD(-a) - CD(a)|


def read_polar(path: str) -> np.ndarray:
    """Read an XFOIL PACC polar file -> array with named columns."""
    data = np.loadtxt(path, skiprows=12)
    return np.atleast_2d(data)


def read_cp(path: str):
    """Read an XFOIL CPWR file -> (x, y, cp) arrays in panel-node order."""
    data = np.loadtxt(path, skiprows=3)
    return data[:, 0], data[:, 1], data[:, 2]


def split_surfaces(x: np.ndarray, y: np.ndarray, cp: np.ndarray):
    """Split a Selig-ordered (TE -> LE -> TE) trace into upper and lower.

    Returns (x_up, cp_up, x_lo, cp_lo), each ordered LE -> TE.
    """
    le = int(np.argmin(x))
    x_up, cp_up = x[: le + 1][::-1], cp[: le + 1][::-1]
    x_lo, cp_lo = x[le:], cp[le:]
    return x_up, cp_up, x_lo, cp_lo


def check(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    return passed


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val_dir", default=os.path.join(here, "xfoil_validation"))
    args = parser.parse_args()

    polar_path = os.path.join(args.val_dir, "naca0012_polar.txt")
    cp0_path = os.path.join(args.val_dir, "cp_alpha0.txt")
    cp4_path = os.path.join(args.val_dir, "cp_alpha4.txt")

    polar = read_polar(polar_path)
    alpha, cl, cd, cdp, cm = polar[:, 0], polar[:, 1], polar[:, 2], polar[:, 3], polar[:, 4]
    top_xtr, bot_xtr = polar[:, 5], polar[:, 6]

    print(f"NACA 0012 validation  (Re=1e6, M=0.15, {len(alpha)} polar points "
          f"from {alpha.min():.0f} to {alpha.max():.0f} deg)\n")

    all_ok = True

    # --- Check 1: CL at alpha = 0 --------------------------------------
    i0 = int(np.argmin(np.abs(alpha)))
    cl0 = cl[i0]
    all_ok &= check("CL(alpha=0) == 0 (symmetric airfoil)",
                    abs(cl0) < TOL_CL_ZERO,
                    f"CL = {cl0:+.6f}  (tol {TOL_CL_ZERO})")

    # CM should also vanish at alpha=0 for a symmetric section.
    all_ok &= check("CM(alpha=0) == 0",
                    abs(cm[i0]) < 1e-3,
                    f"CM = {cm[i0]:+.6f}")

    # --- Check 2: Cp symmetry at alpha = 0 ------------------------------
    x, y, cp = read_cp(cp0_path)
    x_up, cp_up, x_lo, cp_lo = split_surfaces(x, y, cp)
    # Compare on a common grid (surfaces share x by construction for a
    # NACA section, but interpolate to be safe).
    xq = np.linspace(0.0, 1.0, 400)
    cp_up_i = np.interp(xq, x_up, cp_up)
    cp_lo_i = np.interp(xq, x_lo, cp_lo)
    max_cp_diff = float(np.max(np.abs(cp_up_i - cp_lo_i)))
    all_ok &= check("Cp upper == Cp lower at alpha=0",
                    max_cp_diff < TOL_CP_SYM,
                    f"max|Cp_up - Cp_lo| = {max_cp_diff:.3e}  (tol {TOL_CP_SYM})")

    # --- Check 3: polar antisymmetry ------------------------------------
    max_cl_err = 0.0
    max_cd_err = 0.0
    max_xtr_err = 0.0
    for a in alpha[alpha > 0]:
        ip = np.where(np.isclose(alpha, a))[0]
        im = np.where(np.isclose(alpha, -a))[0]
        if len(ip) == 0 or len(im) == 0:
            continue
        ip, im = ip[0], im[0]
        max_cl_err = max(max_cl_err, abs(cl[ip] + cl[im]))
        max_cd_err = max(max_cd_err, abs(cd[ip] - cd[im]))
        max_xtr_err = max(max_xtr_err, abs(top_xtr[ip] - bot_xtr[im]))

    all_ok &= check("CL(-a) == -CL(a)", max_cl_err < TOL_POLAR_ANTISYM,
                    f"max|CL(a)+CL(-a)| = {max_cl_err:.3e}")
    all_ok &= check("CD(-a) == CD(a)", max_cd_err < TOL_POLAR_ANTISYM,
                    f"max|CD(a)-CD(-a)| = {max_cd_err:.3e}")
    all_ok &= check("Top_Xtr(a) == Bot_Xtr(-a)", max_xtr_err < 1e-3,
                    f"max mismatch = {max_xtr_err:.3e}")

    # --- Physical reference check: lift-curve slope ----------------------
    # Thin-airfoil theory gives dCL/dalpha = 2*pi /rad = 0.1097 /deg.
    lin = np.abs(alpha) <= 4
    slope = np.polyfit(alpha[lin], cl[lin], 1)[0]
    all_ok &= check("Lift-curve slope ~ 2*pi/rad (0.1097/deg)",
                    abs(slope - 0.1097) < 0.01,
                    f"dCL/dalpha = {slope:.4f} /deg")

    # --- Plots -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for path, ttl, ax in ((cp0_path, "alpha = 0 deg", axes[0]),
                           (cp4_path, "alpha = 4 deg", axes[1])):
        xx, yy, cc = read_cp(path)
        xu, cu, xl, cl_s = split_surfaces(xx, yy, cc)
        ax.plot(xu, cu, label="upper", lw=1.5)
        ax.plot(xl, cl_s, label="lower", lw=1.5, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_ylabel("$C_p$")
        ax.set_title(f"NACA 0012 $C_p$, {ttl}")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(alpha, cl, "o-", lw=1.5, ms=4)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel("$C_L$")
    ax.set_title("NACA 0012 lift curve (Re=1e6, M=0.15)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.val_dir, "naca0012_validation.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved validation plot -> {out_png}")

    print("\n" + ("ALL VALIDATION CHECKS PASSED" if all_ok
                   else "*** VALIDATION FAILED - do not trust the Step 4 batch run ***"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
