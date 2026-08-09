"""Download the real baseline airfoils from the UIUC Airfoil Coordinate Database.

Source: https://m-selig.ae.illinois.edu/ads/coord_database.html
Files land in data_generation/baselines/ as <canonical_name>.dat and ARE
committed - they're small text inputs to the pipeline, and pinning them
keeps airfoil generation reproducible even if the upstream site changes.

Three requested airfoils use different UIUC filenames (same airfoils,
different naming convention) - see UIUC_FILENAME below.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request

BASE_URL = "https://m-selig.ae.illinois.edu/ads/coord/{}.dat"

# canonical name -> UIUC filename stem
UIUC_FILENAME = {
    "naca0006": "naca0006",
    "naca0012": "n0012",      # substitution: UIUC has NACA 0012 as n0012
    "naca0018": "naca0018",
    "naca2412": "naca2412",
    "naca4412": "naca4412",
    "naca6409": "n6409",      # substitution: UIUC has NACA 6409 as n6409
    "naca23012": "naca23012",
    "clarky": "clarky",
    "e387": "e387",
    "s1223": "s1223",
    "goe387": "goe387",
    "naca4415": "naca4415",
    "naca63-412": "n63412",   # substitution: UIUC has NACA 63-412 as n63412
    "rae2822": "rae2822",
    "sd7062": "sd7062",
}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=os.path.join(here, "baselines"))
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    failures = []

    for canonical, stem in UIUC_FILENAME.items():
        dest = os.path.join(args.out_dir, f"{canonical}.dat")
        if os.path.exists(dest) and not args.force:
            print(f"  skip (exists)  {canonical}")
            continue
        url = BASE_URL.format(stem)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
        except Exception as e:
            print(f"  FAILED         {canonical}  ({url}): {e}")
            failures.append(canonical)
            continue
        with open(dest, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()[:16]
        note = "" if stem == canonical else f"  [UIUC name: {stem}]"
        print(f"  ok  {canonical:12s} {len(data):6d} B  sha256:{sha}{note}")

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}")
        return 1
    print(f"\n{len(UIUC_FILENAME)} baselines in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
