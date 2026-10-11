"""Spread of the Stage 1 results over seeds (42, 43, 44).

Reads overnight/stage1_results.json (seed 42) and stage1_results_s<seed>.json,
writes overnight/stage1_seed_spread.md. Three seeds is enough to show whether
the headline numbers are stable, not enough for tight confidence intervals, so
the spread is reported as min / mean / max alongside the standard deviation.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV = os.path.join(ROOT, "overnight")
SPLITS = ["val", "test", "test_unseen_family"]
METRICS = [("cp_rmse", "Cp RMSE", 4), ("cp_rel_l2", "Cp relative L2", 4), ("cl_mae", "CL MAE (from Cp)", 4)]


def main() -> None:
    runs = {}
    for name, fn in (("42", "stage1_results.json"), ("43", "stage1_results_s43.json"),
                     ("44", "stage1_results_s44.json")):
        p = os.path.join(OV, fn)
        if os.path.exists(p):
            runs[name] = json.load(open(p))
    seeds = list(runs)
    md = ["# Stage 1 spread over seeds", "",
          f"Seeds run: {', '.join(seeds)} ({len(seeds)} of 3 planned). Same data, split and code; only the "
          "initialisation and batch order change.", "",
          "| seed | epochs | best epoch | stop reason | minutes |", "|---|---|---|---|---|"]
    for s in seeds:
        r = runs[s]
        md.append(f"| {s} | {r['epochs_run']} | {r['best_epoch']} | {r['stop_reason']} | {r['minutes']:.1f} |")
    md.append("")
    for sp in SPLITS:
        md += [f"## {sp}", "", "| metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std | min to max |",
               "|---|" + "---|" * (len(seeds) + 3)]
        for key, label, nd in METRICS:
            v = np.array([runs[s]["results"][sp][key] for s in seeds])
            md.append(f"| {label} | " + " | ".join(f"{x:.{nd}f}" for x in v)
                      + f" | {v.mean():.{nd}f} | {v.std(ddof=1) if len(v) > 1 else float('nan'):.{nd}f} | "
                      f"{v.min():.{nd}f} to {v.max():.{nd}f} |")
        md.append("")
    out = os.path.join(OV, "stage1_seed_spread.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
