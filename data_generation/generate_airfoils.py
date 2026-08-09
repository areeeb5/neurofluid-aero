"""Generate the airfoil dataset as perturbations of real baseline airfoils.

WHY THIS APPROACH: the original version sampled CST coefficients
independently from wide fixed ranges. Nothing in that scheme guaranteed
the result was *aerodynamically* well-behaved - the validity filter only
checked self-intersection and a thickness band. In particular the first
Bernstein coefficient sets the leading-edge radius, and sampling it near
zero produces a cusped, zero-radius nose that XFOIL's viscous solver
cannot resolve (it fails even at alpha = 0, not just near stall).

Instead, each airfoil here is a small multiplicative perturbation of CST
coefficients fitted to a real UIUC airfoil. Multiplicative noise
preserves each coefficient's sign and magnitude, so a healthy
leading-edge radius is inherited from the real baseline by construction
rather than hoped for and filtered afterwards.

The geometric validity filter is retained unchanged as a safety net.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cst_airfoil import (
    coordinates_from_cst,
    cosine_spacing,
    generate_airfoil_coordinates,
    is_valid_airfoil,
    thickness_and_camber,
    write_dat_file,
)
from cst_fit import fit_airfoil_cst
from download_baselines import UIUC_FILENAME

MIN_THICKNESS = 0.02
MAX_THICKNESS = 0.22

BASELINES = list(UIUC_FILENAME.keys())


# ---------------------------------------------------------------------------
# Validity-filter unit tests -- unchanged, still run before any real data.
# ---------------------------------------------------------------------------

def run_validity_tests() -> None:
    print("Running validity-filter unit tests...")
    x = cosine_spacing(50)

    y_upper_cross = np.full_like(x, 0.05)
    y_lower_cross = np.full_like(x, -0.05)
    y_lower_cross[20:30] = 0.08
    assert is_valid_airfoil(x, y_upper_cross, y_lower_cross) is False, \
        "FAIL: crossing shape was accepted"
    print("  [PASS] crossing shape correctly rejected")

    y_upper_thin = np.full_like(x, 0.0025)
    y_lower_thin = np.full_like(x, -0.0025)
    assert is_valid_airfoil(x, y_upper_thin, y_lower_thin) is False, \
        "FAIL: too-thin shape was accepted"
    print("  [PASS] too-thin shape correctly rejected")

    y_upper_thick = np.full_like(x, 0.15)
    y_lower_thick = np.full_like(x, -0.15)
    assert is_valid_airfoil(x, y_upper_thick, y_lower_thick) is False, \
        "FAIL: too-thick shape was accepted"
    print("  [PASS] too-thick shape correctly rejected")

    normal_upper = np.array([0.17, 0.16, 0.15, 0.14, 0.13])
    normal_lower = np.array([-0.14, -0.13, -0.12, -0.11, -0.10])
    x_n, y_up, y_lo, _ = generate_airfoil_coordinates(normal_upper, normal_lower,
                                                       n_points=50)
    assert is_valid_airfoil(x_n, y_up, y_lo) is True, \
        "FAIL: normal shape was rejected"
    print("  [PASS] normal shape correctly accepted")

    print("All validity-filter unit tests passed.\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def variant_counts(n_total: int, n_baselines: int):
    """Split n_total as evenly as possible across the baselines."""
    base, rem = divmod(n_total, n_baselines)
    return [base + (1 if i < rem else 0) for i in range(n_baselines)]


def fit_baselines(baselines_dir: str, n_coeffs: int):
    fits = {}
    print(f"Fitting {n_coeffs} CST coefficients/surface to "
          f"{len(BASELINES)} real baselines...")
    for nm in BASELINES:
        f = fit_airfoil_cst(os.path.join(baselines_dir, nm + ".dat"), n_coeffs)
        fits[nm] = f
        print(f"  {nm:14s} max fit err {f['max_err']:.6f} c")
    print()
    return fits


def generate_dataset(args) -> None:
    rng = np.random.default_rng(args.seed)
    fits = fit_baselines(args.baselines_dir, args.n_coeffs)

    os.makedirs(args.output_dir, exist_ok=True)
    for old in os.listdir(args.output_dir):
        if old.endswith(".dat"):
            os.remove(os.path.join(args.output_dir, old))

    counts = variant_counts(args.n_airfoils, len(BASELINES))
    records = []
    accepted = 0
    attempts = 0
    per_baseline_attempts = {}

    for nm, n_want in zip(BASELINES, counts):
        f = fits[nm]
        made = 0
        b_attempts = 0
        max_attempts = n_want * args.max_attempt_factor
        while made < n_want:
            if b_attempts >= max_attempts:
                raise RuntimeError(
                    f"{nm}: only {made}/{n_want} variants accepted after "
                    f"{b_attempts} attempts. The filter is rejecting nearly "
                    f"everything from this baseline - investigate the fit "
                    f"rather than raising the attempt cap.")
            b_attempts += 1
            attempts += 1
            up = f["upper_coeffs"] * (1.0 + rng.normal(0.0, args.perturb,
                                                       size=args.n_coeffs))
            lo = f["lower_coeffs"] * (1.0 + rng.normal(0.0, args.perturb,
                                                       size=args.n_coeffs))
            x, y_upper, y_lower, coords = coordinates_from_cst(
                up, lo, f["te_upper"], f["te_lower"], n_points=args.n_points)

            if not is_valid_airfoil(x, y_upper, y_lower, MIN_THICKNESS,
                                    MAX_THICKNESS):
                continue

            accepted += 1
            made += 1
            airfoil_id = f"airfoil_{accepted:04d}"
            filename = f"{airfoil_id}.dat"
            write_dat_file(os.path.join(args.output_dir, filename), coords,
                           airfoil_id)

            max_t, max_c = thickness_and_camber(x, y_upper, y_lower)
            rec = {"airfoil_id": airfoil_id, "filename": filename,
                   "baseline": nm, "variant_index": made - 1}
            rec.update({f"upper_{i}": up[i] for i in range(args.n_coeffs)})
            rec.update({f"lower_{i}": lo[i] for i in range(args.n_coeffs)})
            rec["te_upper"] = f["te_upper"]
            rec["te_lower"] = f["te_lower"]
            rec["max_thickness"] = max_t
            rec["max_camber"] = max_c
            records.append(rec)
        per_baseline_attempts[nm] = (n_want, b_attempts)

    with open(args.index_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("Per-baseline generation (accepted / attempted):")
    for nm, (n_want, b_att) in per_baseline_attempts.items():
        print(f"  {nm:14s} {n_want:3d} / {b_att:3d}   "
              f"({n_want / b_att * 100:.1f}%)")

    rate = accepted / attempts * 100
    print(f"\nGenerated {accepted} valid airfoils from {attempts} attempts "
          f"({rate:.1f}% acceptance rate).")
    print(f"  perturbation: {args.perturb * 100:.0f}% multiplicative, "
          f"{args.n_coeffs} coefficients/surface")
    print(f"Airfoil .dat files -> {args.output_dir}")
    print(f"Index CSV -> {args.index_csv}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_airfoils", type=int, default=500)
    p.add_argument("--n_coeffs", type=int, default=5,
                   help="CST coefficients per surface. 5 beat 8 on XFOIL "
                        "convergence (95.1%% vs 93.6%%) - extra coefficients "
                        "give the perturbation more room to make the surface "
                        "wavy, which costs more than the better fit gains.")
    p.add_argument("--perturb", type=float, default=0.10,
                   help="multiplicative coefficient noise (std, fraction)")
    p.add_argument("--n_points", type=int, default=100,
                   help="points per surface (cosine-spaced)")
    p.add_argument("--max_attempt_factor", type=int, default=50,
                   help="abort a baseline after n_want * this many rejections")
    p.add_argument("--baselines_dir", default=os.path.join(here, "baselines"))
    p.add_argument("--output_dir", default=os.path.join(here, "airfoils"))
    p.add_argument("--index_csv", default=os.path.join(here, "airfoil_index.csv"))
    args = p.parse_args()

    run_validity_tests()
    generate_dataset(args)


if __name__ == "__main__":
    main()
