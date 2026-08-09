"""Batch-generate CST-parameterized airfoils for the XFOIL data pipeline.

Randomly samples CST coefficients (upper coefficients biased positive,
lower biased negative/small, so draws land near realistic shapes),
filters out geometrically invalid candidates, and writes exactly
--n_airfoils valid airfoils plus a master index CSV. See data_spec.md at
the repo root for the output format contract.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cst_airfoil import (
    N_COEFFS,
    cosine_spacing,
    generate_airfoil_coordinates,
    is_valid_airfoil,
    thickness_and_camber,
    write_dat_file,
)

MIN_THICKNESS = 0.02
MAX_THICKNESS = 0.22

# Sampling distributions: upper coefficients biased positive, lower biased
# negative/small, so random draws land near realistic shapes rather than
# relying only on the validity filter to reject noise.
UPPER_MEAN, UPPER_STD, UPPER_MIN = 0.15, 0.18, 0.0
LOWER_MEAN, LOWER_STD, LOWER_MAX = -0.10, 0.15, 0.0


def sample_coefficients(rng: np.random.Generator):
    upper = np.clip(rng.normal(UPPER_MEAN, UPPER_STD, size=N_COEFFS), UPPER_MIN, None)
    lower = np.clip(rng.normal(LOWER_MEAN, LOWER_STD, size=N_COEFFS), None, LOWER_MAX)
    return upper, lower


# ---------------------------------------------------------------------------
# Validity-filter unit tests -- must pass before any real data is generated.
# ---------------------------------------------------------------------------

def run_validity_tests() -> None:
    print("Running validity-filter unit tests...")
    x = cosine_spacing(50)

    # 1. Deliberately crossing shape: part of the "lower" surface pokes
    #    above the "upper" surface.
    y_upper_cross = np.full_like(x, 0.05)
    y_lower_cross = np.full_like(x, -0.05)
    y_lower_cross[20:30] = 0.08
    assert is_valid_airfoil(x, y_upper_cross, y_lower_cross) is False, \
        "FAIL: crossing shape was accepted"
    print("  [PASS] crossing shape correctly rejected")

    # 2. Deliberately too-thin shape (uniform 0.5% chord thickness).
    y_upper_thin = np.full_like(x, 0.0025)
    y_lower_thin = np.full_like(x, -0.0025)
    assert is_valid_airfoil(x, y_upper_thin, y_lower_thin) is False, \
        "FAIL: too-thin shape was accepted"
    print("  [PASS] too-thin shape correctly rejected")

    # 3. Deliberately too-thick shape (uniform 30% chord thickness).
    y_upper_thick = np.full_like(x, 0.15)
    y_lower_thick = np.full_like(x, -0.15)
    assert is_valid_airfoil(x, y_upper_thick, y_lower_thick) is False, \
        "FAIL: too-thick shape was accepted"
    print("  [PASS] too-thick shape correctly rejected")

    # 4. Normal shape: a plain CST airfoil at ~12% thickness should pass.
    normal_upper = np.array([0.17, 0.16, 0.15, 0.14, 0.13])
    normal_lower = np.array([-0.14, -0.13, -0.12, -0.11, -0.10])
    x_n, y_up, y_lo, _ = generate_airfoil_coordinates(normal_upper, normal_lower, n_points=50)
    assert is_valid_airfoil(x_n, y_up, y_lo) is True, \
        "FAIL: normal shape was rejected"
    print("  [PASS] normal shape correctly accepted")

    print("All validity-filter unit tests passed.\n")


# ---------------------------------------------------------------------------
# Batch generation.
# ---------------------------------------------------------------------------

def generate_dataset(n_airfoils: int, seed: int, output_dir: str, index_csv: str,
                      n_points: int, te_thickness: float) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    accepted = 0
    attempts = 0

    while accepted < n_airfoils:
        attempts += 1
        upper, lower = sample_coefficients(rng)
        x, y_upper, y_lower, coords = generate_airfoil_coordinates(
            upper, lower, n_points=n_points, te_thickness=te_thickness
        )
        if not is_valid_airfoil(x, y_upper, y_lower, MIN_THICKNESS, MAX_THICKNESS):
            continue

        accepted += 1
        airfoil_id = f"airfoil_{accepted:04d}"
        filename = f"{airfoil_id}.dat"
        write_dat_file(os.path.join(output_dir, filename), coords, airfoil_id)

        max_t, max_c = thickness_and_camber(x, y_upper, y_lower)
        record = {"airfoil_id": airfoil_id, "filename": filename}
        record.update({f"upper_{i}": upper[i] for i in range(N_COEFFS)})
        record.update({f"lower_{i}": lower[i] for i in range(N_COEFFS)})
        record["max_thickness"] = max_t
        record["max_camber"] = max_c
        records.append(record)

    fieldnames = list(records[0].keys())
    with open(index_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    acceptance_rate = accepted / attempts * 100
    print(f"Generated {accepted} valid airfoils from {attempts} attempts "
          f"({acceptance_rate:.1f}% acceptance rate).")
    print(f"Airfoil .dat files -> {output_dir}")
    print(f"Index CSV -> {index_csv}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_airfoils", type=int, default=500)
    parser.add_argument("--n_points", type=int, default=100,
                         help="points per surface (cosine-spaced)")
    parser.add_argument("--te_thickness", type=float, default=0.002,
                         help="trailing-edge thickness, chord fraction (0 = sharp TE)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(here, "airfoils"))
    parser.add_argument("--index_csv", type=str, default=os.path.join(here, "airfoil_index.csv"))
    args = parser.parse_args()

    run_validity_tests()

    generate_dataset(
        n_airfoils=args.n_airfoils,
        seed=args.seed,
        output_dir=args.output_dir,
        index_csv=args.index_csv,
        n_points=args.n_points,
        te_thickness=args.te_thickness,
    )


if __name__ == "__main__":
    main()
