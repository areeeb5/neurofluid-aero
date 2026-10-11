"""Classical-ML baselines (Phase 1 prep for the Week 2 "baselines" item).

Predict the three headline coefficients (CL, CD, CM) from the airfoil's CST
description plus the angle of attack. This is the cheap benchmark the neural
operator has to beat, in the spirit of Teimourian et al. 2024 (RF / GBR) and
Moin et al. 2021, who found CD the hardest target (lowest R^2).

Trained on the `train` split only. Scored on val, test and, most importantly,
`test_unseen_family` (whole baseline families never seen in training).
No hyperparameter tuning, so the val split is reported but not used to choose.
Fully reproducible from --seed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["CL", "CD", "CM"]
SPLITS = ["val", "test", "test_unseen_family"]


def load(index_csv, splits_csv, polars_csv):
    with open(index_csv, newline="") as f:
        idx = {r["airfoil_id"]: r for r in csv.DictReader(f)}
    with open(splits_csv, newline="") as f:
        split = {r["airfoil_id"]: r["split"] for r in csv.DictReader(f)}
    feat_cols = ([f"upper_{i}" for i in range(5)] + [f"lower_{i}" for i in range(5)]
                 + ["te_upper", "te_lower"])
    X, Y, S = [], [], []
    with open(polars_csv, newline="") as f:
        for r in csv.DictReader(f):
            a = idx[r["airfoil_id"]]
            X.append([float(a[c]) for c in feat_cols] + [float(r["alpha"])])
            Y.append([float(r[t]) for t in TARGETS])
            S.append(split[r["airfoil_id"]])
    return np.array(X), np.array(Y), np.array(S), feat_cols + ["alpha"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=2)
    p.add_argument("--out_dir", default=os.path.join(HERE, "baselines_results"))
    args = p.parse_args()

    X, Y, S, cols = load(os.path.join(HERE, "airfoil_index.csv"),
                         os.path.join(HERE, "splits.csv"),
                         os.path.join(HERE, "xfoil_data", "polars.csv"))
    tr = S == "train"
    print(f"features {len(cols)}; train cases {tr.sum()}; "
          + "; ".join(f"{s} {int((S == s).sum())}" for s in SPLITS))

    models = {
        "mean (no skill)": lambda: DummyRegressor(strategy="mean"),
        "linear": lambda: LinearRegression(),
        "random forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=args.seed, n_jobs=args.n_jobs),
        "gradient boosting": lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=args.seed),
    }

    rows = []
    for mname, make in models.items():
        for ti, t in enumerate(TARGETS):
            m = make().fit(X[tr], Y[tr, ti])
            for s in SPLITS:
                te = S == s
                pred = m.predict(X[te])
                rows.append({"model": mname, "target": t, "split": s,
                             "R2": r2_score(Y[te, ti], pred),
                             "MAE": mean_absolute_error(Y[te, ti], pred),
                             "target_std": float(Y[te, ti].std())})
        print(f"  trained {mname}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "classical_baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def cell(model, target, split, key):
        return next(r[key] for r in rows if r["model"] == model and r["target"] == target
                    and r["split"] == split)

    md = ["# Classical-ML baselines", "",
          "Inputs: 10 CST coefficients + 2 trailing-edge offsets + angle of attack. "
          f"Trained on {int(tr.sum())} cases from the `train` split. Seed {args.seed}. "
          "R2 = 1 is perfect, 0 is no better than predicting the training mean.", ""]
    for s in SPLITS:
        md += [f"## {s}", "", "| model | R2 CL | R2 CD | R2 CM | MAE CL | MAE CD | MAE CM |",
               "|---|---|---|---|---|---|---|"]
        for mname in models:
            md.append("| " + mname + " | "
                      + " | ".join(f"{cell(mname, t, s, 'R2'):.3f}" for t in TARGETS) + " | "
                      + " | ".join(f"{cell(mname, t, s, 'MAE'):.4f}" for t in TARGETS) + " |")
        md.append("")
    best = "random forest"
    r2s = {t: cell(best, t, "test", "R2") for t in TARGETS}
    hardest = min(r2s, key=r2s.get)
    md += ["## Reading", "",
           f"- On the in-family `test` split the random forest's lowest R2 is **{hardest}** "
           f"({r2s[hardest]:.3f}); Moin et al. 2021 found CD the hardest target.",
           "- Compare `test` against `test_unseen_family`: the gap is how much of the "
           "skill is memorising familiar baseline families rather than learning aerodynamics.",
           "- The dataset is 15 tight clusters around real airfoils, so in-family splits "
           "are close to interpolation and these baselines score very high there. "
           "Whether the neural operator (Stage 1) beats them is an open question; the "
           "`test_unseen_family` split is the meaningful bar."]
    with open(os.path.join(args.out_dir, "classical_baselines.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
