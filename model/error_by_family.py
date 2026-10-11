"""Stage 1 error broken down by baseline family and by thickness.

Reads overnight/stage1_per_airfoil_error.csv (seed 42 run) and writes
overnight/stage1_error_by_family.md. Answers: is the error concentrated in
particular families, and does it grow for thin sections?
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    rows = list(csv.DictReader(open(os.path.join(ROOT, "overnight", "stage1_per_airfoil_error.csv"))))
    idx = {a["airfoil_id"]: a for a in csv.DictReader(
        open(os.path.join(ROOT, "data_generation", "airfoil_index.csv")))}
    by = defaultdict(list)
    for x in rows:
        by[x["baseline"]].append(x)

    table = []
    for b, xs in by.items():
        th = np.mean([float(idx[x["airfoil_id"]]["max_thickness"]) for x in xs]) * 100
        tr = [float(x["mean_cp_rel_l2"]) for x in xs if x["split"] == "train"]
        ho = [float(x["mean_cp_rel_l2"]) for x in xs if x["split"] != "train"]
        table.append((b, th, len(xs), np.mean(tr) if tr else None, np.mean(ho) if ho else None))
    table.sort(key=lambda t: t[1])

    vt = [x for x in rows if x["split"] in ("val", "test")]
    t = np.array([float(idx[x["airfoil_id"]]["max_thickness"]) for x in vt])
    e = np.array([float(x["mean_cp_rel_l2"]) for x in vt])
    corr = float(np.corrcoef(t, e)[0, 1])

    f = lambda v: "n/a (held out)" if v is None else f"{v*100:.1f}%"
    md = ["# Stage 1 error by baseline family (seed 42)", "",
          "Mean relative Cp error (L2) per airfoil, averaged over a family. "
          "`train` airfoils were fitted by the model; `held-out` = val + test + unseen-family airfoils.", "",
          "| baseline | mean thickness | airfoils | train error | held-out error |", "|---|---|---|---|---|"]
    md += [f"| {b} | {th:.1f}% | {n} | {f(tr)} | {f(ho)} |" for b, th, n, tr, ho in table]
    md += ["", f"Correlation of airfoil error with thickness over the {len(vt)} val+test airfoils: **{corr:+.2f}** "
           "(negative = thinner airfoils have larger error).", "",
           "## Reading", "",
           "- The two held-out families (`naca6409`, `e387`) have about 17 to 18% error against 2 to 5% "
           "for families the model trained on: generalising to an unseen family is the main gap.",
           "- `naca0006`, the thinnest family, has roughly twice the error of the others even on its "
           "TRAINING airfoils. This is consistent with, but does not prove, thin sections being "
           "resolved coarsely by the 64x64 grid (their mask is only 2 cells thick).",
           "- Separately, a single-case diagnostic (`overfit_single_sample.py --hardest`) shows the "
           "network can memorise even the sharpest suction spike in the data (Cp down to -8.84) to "
           "0.3% error, so the spike errors on unseen cases are not a capacity limit of the network."]
    out = os.path.join(ROOT, "overnight", "stage1_error_by_family.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
