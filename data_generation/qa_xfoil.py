"""QA of the Step 4 XFOIL results. Flags suspicious converged cases, deletes nothing.

Why: "converged" only means XFOIL reached a stored polar point; it does not
mean the result is physical. The Cp curves are the Stage 1 training target, so
they are cross-checked against the reported coefficients.

Checks (HARD = physically impossible or corrupt, REVIEW = unusual, look at it):
  HARD   non_finite       NaN / inf anywhere in the case
  HARD   cd_range         CD <= 0 or CD > 0.5
  HARD   cl_range         |CL| > 3.5
  HARD   xtr_range        Top_Xtr or Bot_Xtr outside [0, 1]
  HARD   cp_range         Cp max > 1.02 or Cp min < -30
  HARD   cd_lt_cdp        total CD below pressure drag CDp (by more than 1e-4)
  HARD   cl_cp_mismatch   |CL from integrating the Cp curve - reported CL| > 0.15
  REVIEW cl_cp_mismatch   0.05 < that difference <= 0.15
  REVIEW cl_kink          |second difference of CL| > 0.30 between 1-degree steps
  REVIEW cd_jump          CD changes by more than 2.5x between adjacent 1-degree steps
  REVIEW lift_slope       slope dCL/dalpha < -0.02 per degree while |alpha| <= 8

Cp integration (CCW traversal, Selig order TE -> upper -> LE -> lower -> TE):
  Cy = sum 0.5 (Cp_i + Cp_j)(x_j - x_i),  Cx = -sum 0.5 (Cp_i + Cp_j)(y_j - y_i),
  CL = Cy cos(a) - Cx sin(a). Sign and form are verified against the NACA 0012
  validation case by --selftest before the dataset is checked.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cl_from_cp(x, y, cp, alpha_deg):
    """Lift coefficient by integrating Cp around the closed outline."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cp = np.asarray(cp, float)
    j = np.roll(np.arange(len(x)), -1)  # closing segment included
    cpm = 0.5 * (cp + cp[j])
    cy = np.sum(cpm * (x[j] - x))
    cx = -np.sum(cpm * (y[j] - y))
    a = np.deg2rad(alpha_deg)
    return cy * np.cos(a) - cx * np.sin(a), cy * np.sin(a) + cx * np.cos(a)


def selftest() -> bool:
    """NACA 0012 at 4 degrees: Cp integral must reproduce the polar's CL."""
    vdir = os.path.join(HERE, "xfoil_validation")
    cp_file = os.path.join(vdir, "cp_alpha4.txt")
    polar = os.path.join(vdir, "naca0012_polar.txt")
    d = np.loadtxt(cp_file, skiprows=3)
    cl_ref = None
    for line in open(polar).read().splitlines()[12:]:
        p = line.split()
        if len(p) >= 7 and abs(float(p[0]) - 4.0) < 1e-6:
            cl_ref = float(p[1])
    cl, cdp = cl_from_cp(d[:, 0], d[:, 1], d[:, 2], 4.0)
    ok = cl_ref is not None and abs(cl - cl_ref) < 0.02
    print(f"selftest NACA0012 @4deg: CL from Cp {cl:.4f} vs polar {cl_ref}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    # a deliberately wrong sign convention must NOT pass
    wrong, _ = cl_from_cp(d[::-1, 0], d[::-1, 1], d[::-1, 2], 4.0)
    bad_ok = abs(wrong - cl_ref) > 0.1
    print(f"selftest reversed traversal gives {wrong:.4f} (must differ from {cl_ref}) "
          f"-> {'PASS' if bad_ok else 'FAIL'}")
    return ok and bad_ok


def check_airfoil(aid, base, d):
    """Run every check on one airfoil's converged cases (d = dict of arrays).
    Returns (flags, diffs, n_cases)."""
    flags, diffs = [], []
    al, cl, cd, cdp = d["alpha"], d["CL"], d["CD"], d["CDp"]
    top, bot, cp = d["Top_Xtr"], d["Bot_Xtr"], d["cp"]
    x, y = d["x"], d["y"]

    def add(i, sev, chk, val):
        flags.append((aid, base, f"{al[i]:g}", sev, chk, f"{val:.6g}"))

    for i in range(len(al)):
        row = np.concatenate([[cl[i], cd[i], cdp[i], top[i], bot[i]], cp[i]])
        if not np.all(np.isfinite(row)):
            add(i, "HARD", "non_finite", np.nan)
            continue
        if cd[i] <= 0 or cd[i] > 0.5:
            add(i, "HARD", "cd_range", cd[i])
        if abs(cl[i]) > 3.5:
            add(i, "HARD", "cl_range", cl[i])
        if not (0 <= top[i] <= 1 and 0 <= bot[i] <= 1):
            add(i, "HARD", "xtr_range", max(top[i], bot[i]))
        if cp[i].max() > 1.02 or cp[i].min() < -30:
            add(i, "HARD", "cp_range", cp[i].max() if cp[i].max() > 1.02 else cp[i].min())
        if cd[i] < cdp[i] - 1e-4:
            add(i, "HARD", "cd_lt_cdp", cd[i] - cdp[i])
        clc, _ = cl_from_cp(x, y, cp[i], al[i])
        diff = clc - cl[i]
        diffs.append((aid, base, al[i], diff))
        if abs(diff) > 0.15:
            add(i, "HARD", "cl_cp_mismatch", diff)
        elif abs(diff) > 0.05:
            add(i, "REVIEW", "cl_cp_mismatch", diff)

    # neighbour checks on adjacent 1-degree steps only
    for i in range(1, len(al) - 1):
        if abs(al[i] - al[i - 1] - 1) < 1e-6 and abs(al[i + 1] - al[i] - 1) < 1e-6:
            d2 = cl[i + 1] - 2 * cl[i] + cl[i - 1]
            if abs(d2) > 0.30:
                add(i, "REVIEW", "cl_kink", d2)
    for i in range(1, len(al)):
        if abs(al[i] - al[i - 1] - 1) < 1e-6:
            r = cd[i] / cd[i - 1] if cd[i - 1] > 0 else np.inf
            if r > 2.5 or r < 0.4:
                add(i, "REVIEW", "cd_jump", r)
            if (cl[i] - cl[i - 1]) < -0.02 and abs(al[i]) <= 8:
                add(i, "REVIEW", "lift_slope", cl[i] - cl[i - 1])
    return flags, diffs, len(al)


def selftest_checks(cp_dir: str) -> bool:
    """Known-good and known-bad: a real case must be clean, and each injected
    fault must trip exactly the check meant to catch it."""
    fn = next(f for f in sorted(os.listdir(cp_dir)) if f.endswith(".npz"))
    raw = np.load(os.path.join(cp_dir, fn), allow_pickle=False)
    good = {k: raw[k].copy() for k in raw.files if k != "baseline"}
    ok = True

    f0, _, _ = check_airfoil("t", "t", good)
    clean = len(f0) == 0
    print(f"selftest checks: real case {fn[:-4]} raises no flags -> {'PASS' if clean else 'FAIL ' + str(f0)}")
    ok &= clean

    def fault(name, mutate, expect):
        d = {k: v.copy() for k, v in good.items()}
        mutate(d)
        got = {f[4] for f in check_airfoil("t", "t", d)[0]}
        hit = expect in got
        print(f"selftest checks: injected {name} trips '{expect}' -> {'PASS' if hit else 'FAIL, got ' + str(got)}")
        return hit

    ok &= fault("negative CD", lambda d: d["CD"].__setitem__(0, -0.01), "cd_range")
    ok &= fault("NaN in Cp", lambda d: d["cp"].__setitem__((0, 5), np.nan), "non_finite")
    ok &= fault("transition at x/c = 1.5", lambda d: d["Top_Xtr"].__setitem__(0, 1.5), "xtr_range")
    ok &= fault("Cp spike of +5", lambda d: d["cp"].__setitem__((0, 5), 5.0), "cp_range")
    ok &= fault("CL shifted by 0.4 (Cp no longer matches)", lambda d: d["CL"].__setitem__(0, d["CL"][0] + 0.4), "cl_cp_mismatch")
    ok &= fault("CD below pressure drag", lambda d: d["CD"].__setitem__(0, d["CDp"][0] - 0.01 if d["CDp"][0] > 0.011 else 1e-6), "cd_lt_cdp")
    return bool(ok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cp_dir", default=os.path.join(HERE, "xfoil_data", "cp"))
    p.add_argument("--out_csv", default=os.path.join(HERE, "xfoil_data", "qa_flags.csv"))
    p.add_argument("--out_md", default=os.path.join(HERE, "xfoil_batch_report", "qa_summary.md"))
    args = p.parse_args()

    if not (selftest() and selftest_checks(args.cp_dir)):
        print("selftest failed, refusing to run QA on the dataset", file=sys.stderr)
        return 1

    flags = []  # (airfoil_id, baseline, alpha, severity, check, value)
    diffs = []
    n_cases = 0
    for fn in sorted(os.listdir(args.cp_dir)):
        if not fn.endswith(".npz"):
            continue
        raw = np.load(os.path.join(args.cp_dir, fn), allow_pickle=False)
        d = {k: raw[k] for k in raw.files if k != "baseline"}
        f, dv_, n = check_airfoil(fn[:-4], str(raw["baseline"]), d)
        flags += f
        diffs += dv_
        n_cases += n

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airfoil_id", "baseline", "alpha", "severity", "check", "value"])
        w.writerows(flags)

    # ---- summary ----
    from collections import Counter
    cnt = Counter((f[3], f[4]) for f in flags)
    hard_cases = {(f[0], f[2]) for f in flags if f[3] == "HARD"}
    rev_cases = {(f[0], f[2]) for f in flags if f[3] == "REVIEW"} - hard_cases
    dv = np.array([t[3] for t in diffs])
    by_base = Counter(f[1] for f in flags if f[3] == "HARD")

    lines = [
        "# XFOIL data QA", "",
        f"{n_cases} converged cases checked. Flags are advisory: nothing was deleted or changed. "
        "Full list: `xfoil_data/qa_flags.csv`.", "",
        "| metric | value |", "|---|---|",
        f"| cases with a HARD flag | {len(hard_cases)} ({len(hard_cases)/n_cases*100:.2f}%) |",
        f"| cases with only REVIEW flags | {len(rev_cases)} ({len(rev_cases)/n_cases*100:.2f}%) |",
        f"| clean cases | {n_cases - len(hard_cases) - len(rev_cases)} "
        f"({(n_cases - len(hard_cases) - len(rev_cases))/n_cases*100:.2f}%) |", "",
        "## Flags by check", "", "| severity | check | count |", "|---|---|---|",
    ]
    for (sev, chk), c in sorted(cnt.items(), key=lambda kv: (kv[0][0], -kv[1])):
        lines.append(f"| {sev} | {chk} | {c} |")
    lines += [
        "", "## Does the Cp curve reproduce the reported CL?", "",
        "CL from integrating the stored Cp curve minus the polar's CL, over every converged case:", "",
        f"- mean {dv.mean():+.4f}, median {np.median(dv):+.4f}, std {dv.std():.4f}",
        f"- 95% of cases within {np.percentile(np.abs(dv), 95):.4f}, "
        f"99% within {np.percentile(np.abs(dv), 99):.4f}, max {np.abs(dv).max():.4f}",
        "- This is a file-integrity check, not a physics validation: XFOIL computes its CL "
        "from this same Cp distribution, so agreement shows the saved Cp curves are "
        "complete, correctly ordered and matched to the right angle (no mix-ups), "
        "not that the flow solution is physically right.", "",
        "## HARD flags by baseline", "",
    ]
    if by_base:
        lines += ["| baseline | HARD flags |", "|---|---|"]
        lines += [f"| {b} | {c} |" for b, c in by_base.most_common()]
    else:
        lines.append("None.")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
