"""DRAFT active selection: which airfoils is the Stage 1 model most wrong about?

For review by Areeb only. Nothing here is sent to ANSYS.

Method: score every converged case with the best Stage 1 checkpoint, rank
airfoils by their mean relative Cp error, and propose the top N with at most
--max_per_family from any one baseline family (so the 15-20 expensive ANSYS
cases are not all copies of one hard family). For each proposed airfoil the
angle with the worst error is suggested as the ANSYS operating point.

Caveats written into the output:
  * Errors on `train` airfoils are optimistic (the model has fitted them), so
    the ranking is shown both over all airfoils and over held-out ones.
  * A large error at high alpha on a thin airfoil may reflect XFOIL being
    unreliable near stall rather than the model failing; check before spending
    ANSYS time there.
Needs overnight/checkpoints/stage1_best.pt (from train_stage1.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fno  # noqa: E402
from data_stage1 import Stage1Data  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "overnight")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_per_family", type=int, default=3)
    p.add_argument("--min_cases", type=int, default=8,
                   help="ignore airfoils with fewer converged angles than this")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Stage1Data(device)
    net = fno.Stage1Net().to(device)
    net.load_state_dict(torch.load(os.path.join(OUT, "checkpoints", "stage1_best.pt"),
                                   map_location=device))
    net.eval()
    with torch.no_grad():
        idx = torch.arange(data.n_cases, device=device)
        preds = []
        for i in range(0, len(idx), 256):
            g_in, sxy, cp, alpha, cl = data.batch(idx[i:i + 256])
            preds.append(net(g_in, sxy))
        pred = torch.cat(preds)
    rel = ((pred - data.case_cp).norm(dim=1) / data.case_cp.norm(dim=1)).cpu().numpy()
    ai = data.case_ai.cpu().numpy()
    alpha = data.case_alpha.cpu().numpy()

    per = defaultdict(list)
    for k in range(data.n_cases):
        per[int(ai[k])].append(k)
    rows = []
    for a, ks in per.items():
        if len(ks) < args.min_cases:
            continue
        aid = data.ids[a]
        worst = ks[int(np.argmax(rel[ks]))]
        rows.append({"airfoil_id": aid, "baseline": data.baseline[aid],
                     "split": data.case_split[ks[0]],
                     "n_cases": len(ks), "mean_cp_rel_l2": float(rel[ks].mean()),
                     "worst_alpha": float(alpha[worst]), "worst_case_rel_l2": float(rel[worst])})

    def pick(pool):
        pool = sorted(pool, key=lambda r: -r["mean_cp_rel_l2"])
        chosen, fam = [], Counter()
        for r in pool:
            if fam[r["baseline"]] >= args.max_per_family:
                continue
            chosen.append(r)
            fam[r["baseline"]] += 1
            if len(chosen) == args.n:
                break
        return chosen

    views = {"all airfoils": rows,
             "held-out airfoils only (val, test, unseen family)": [r for r in rows if r["split"] != "train"]}
    out_rows = []
    md = ["# DRAFT active selection (not sent to ANSYS)", "",
          f"Top {args.n} airfoils by mean relative Cp error of the Stage 1 model, at most "
          f"{args.max_per_family} per baseline family, airfoils with at least {args.min_cases} "
          "converged angles. `worst_alpha` is the suggested ANSYS angle (largest error).", "",
          "Caveats: train-split errors are optimistic. A large error at high alpha on a thin "
          "airfoil may be XFOIL near stall rather than a model failure; check before running ANSYS.", ""]
    for name, pool in views.items():
        chosen = pick(pool)
        md += [f"## {name}", "", "| rank | airfoil | baseline | split | mean Cp rel. L2 | worst alpha | worst-case rel. L2 |",
               "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(chosen, 1):
            md.append(f"| {i} | {r['airfoil_id']} | {r['baseline']} | {r['split']} | "
                      f"{r['mean_cp_rel_l2']:.3f} | {r['worst_alpha']:g} | {r['worst_case_rel_l2']:.3f} |")
            out_rows.append({"view": name, "rank": i, **r})
        md += ["", "Families in this list: " + ", ".join(f"{b} x{c}" for b, c in
               Counter(r["baseline"] for r in chosen).most_common()), ""]
    with open(os.path.join(OUT, "active_selection_draft.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    open(os.path.join(OUT, "active_selection_draft.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
