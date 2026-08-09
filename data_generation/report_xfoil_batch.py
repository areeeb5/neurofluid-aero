"""Visual summary of the Step 4 batch XFOIL run.

Reads xfoil_data/{status.csv, polars.csv, cp/*.npz} and writes to
xfoil_batch_report/:
  convergence_overview.png     overall split + per-airfoil rate, worst to best
  coverage_grid.png            airfoils x angles, colored by status
  cl_alpha_samples.png         CL vs alpha for 10 random airfoils
  convergence_by_baseline.png  convergence rate per real baseline family
  summary.md                   headline numbers
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

STATUSES = ["converged", "failed", "timed_out"]
COLORS = {"converged": "#2e8b57", "failed": "#f0a202", "timed_out": "#b22222"}


def load_status(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_overview(rows, out_path):
    counts = collections.Counter(r["status"] for r in rows)
    total = sum(counts.values())

    per_af = collections.defaultdict(collections.Counter)
    for r in rows:
        per_af[r["airfoil_id"]][r["status"]] += 1
    rates = sorted((c["converged"] / sum(c.values()) * 100)
                   for c in per_af.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [1, 2.4]})

    left = 0.0
    for s in STATUSES:
        v = counts.get(s, 0) / total * 100
        if v <= 0:
            continue
        ax1.barh([0], [v], left=left, color=COLORS[s],
                 label=f"{s} {v:.2f}% ({counts.get(s,0)})")
        left += v
    ax1.set_xlim(0, 100)
    ax1.set_yticks([])
    ax1.set_xlabel("% of all cases")
    ax1.set_title(f"Overall status ({total} cases)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=9)

    ax2.bar(range(len(rates)), rates, width=1.0, color="#2e8b57",
            edgecolor="none")
    ax2.axhline(100, color="0.4", lw=0.8, ls="--")
    mean_rate = float(np.mean(rates))
    ax2.axhline(mean_rate, color="tab:blue", lw=1.2,
                label=f"mean {mean_rate:.1f}%")
    ax2.set_xlabel("airfoil (sorted worst -> best)")
    ax2.set_ylabel("% of angles converged")
    ax2.set_ylim(0, 105)
    ax2.set_title("Per-airfoil convergence rate")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_coverage_grid(rows, out_path):
    """Airfoils (rows) x angles (columns), grouped by baseline."""
    alphas = sorted({float(r["alpha"]) for r in rows})
    a_idx = {a: i for i, a in enumerate(alphas)}
    s_idx = {s: i for i, s in enumerate(STATUSES)}

    by_af = collections.defaultdict(dict)
    baseline_of = {}
    for r in rows:
        by_af[r["airfoil_id"]][float(r["alpha"])] = r["status"]
        baseline_of[r["airfoil_id"]] = r.get("baseline", "")

    # Group by baseline so cluster-level patterns are visible at a glance.
    order = sorted(by_af, key=lambda a: (baseline_of[a], a))
    grid = np.full((len(order), len(alphas)), np.nan)
    for i, aid in enumerate(order):
        for a, s in by_af[aid].items():
            grid[i, a_idx[a]] = s_idx[s]

    fig, ax = plt.subplots(figsize=(9, 12))
    cmap = ListedColormap([COLORS[s] for s in STATUSES])
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(STATUSES) - 0.5,
              interpolation="nearest")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8)
    ax.set_xlabel("angle of attack (deg)")

    # Baseline separators + labels.
    bounds, labels = [], []
    prev, start = None, 0
    for i, aid in enumerate(order):
        b = baseline_of[aid]
        if b != prev:
            if prev is not None:
                bounds.append(i)
                labels.append((start + i) / 2 - 0.5)
            prev, start = b, i
    labels.append((start + len(order)) / 2 - 0.5)
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.1)
    names = []
    prev = None
    for aid in order:
        b = baseline_of[aid]
        if b != prev:
            names.append(b)
            prev = b
    ax.set_yticks(labels)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylabel("airfoil, grouped by real baseline")

    ax.legend(handles=[Patch(color=COLORS[s], label=s) for s in STATUSES],
              loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3)
    ax.set_title("Case coverage: every (airfoil, angle) pair")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cl_alpha(cp_dir, out_path, n=10, seed=42):
    files = sorted(f for f in os.listdir(cp_dir) if f.endswith(".npz"))
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in rng.choice(len(files), size=min(n, len(files)),
                                         replace=False)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for fn in pick:
        d = np.load(os.path.join(cp_dir, fn), allow_pickle=False)
        base = str(d["baseline"]) if "baseline" in d else ""
        ax.plot(d["alpha"], d["CL"], "o-", ms=3.5, lw=1.3,
                label=f"{fn[:-4]} ({base})")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("angle of attack (deg)")
    ax.set_ylabel("$C_L$")
    ax.set_title("Lift curves, 10 random airfoils (physical sanity check)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_by_baseline(rows, out_path):
    per_b = collections.defaultdict(collections.Counter)
    for r in rows:
        per_b[r.get("baseline", "")][r["status"]] += 1

    items = []
    for b, c in per_b.items():
        tot = sum(c.values())
        items.append((b, c["converged"] / tot * 100, tot, c))
    items.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [t[0] for t in items]
    ypos = np.arange(len(items))
    left = np.zeros(len(items))
    for s in STATUSES:
        vals = np.array([t[3][s] / t[2] * 100 for t in items])
        ax.barh(ypos, vals, left=left, color=COLORS[s], label=s)
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("% of cases")
    ax.set_xlim(0, 100)
    ax.set_title("Convergence by real baseline family (sorted worst -> best)")
    for i, t in enumerate(items):
        ax.text(101, i, f"{t[1]:.1f}%", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return items


def write_summary(rows, items, out_path, runtime_s, workers, cp_dir):
    counts = collections.Counter(r["status"] for r in rows)
    total = sum(counts.values())
    n_af = len({r["airfoil_id"] for r in rows})
    n_al = len({r["alpha"] for r in rows})

    size = 0
    for dirpath, _, files in os.walk(os.path.dirname(cp_dir)):
        for fn in files:
            size += os.path.getsize(os.path.join(dirpath, fn))

    lines = [
        "# Step 4 - batch XFOIL run summary", "",
        f"{n_af} airfoils x {n_al} angles of attack "
        f"(-4 deg to 10 deg, 1 deg steps), Re = 1e6, M = 0.15, Ncrit = 9 "
        f"(default), free transition, 200 panel nodes.", "",
        "| metric | value |", "|---|---|",
        f"| total cases | {total} |",
        f"| converged | {counts.get('converged',0)} "
        f"({counts.get('converged',0)/total*100:.2f}%) |",
        f"| failed | {counts.get('failed',0)} "
        f"({counts.get('failed',0)/total*100:.2f}%) |",
        f"| timed out | {counts.get('timed_out',0)} "
        f"({counts.get('timed_out',0)/total*100:.2f}%) |",
    ]
    if runtime_s:
        lines += [
            f"| total runtime | {runtime_s:.1f} s ({runtime_s/60:.1f} min) |",
            f"| parallel workers | {workers} |",
            f"| average wall time / case | {runtime_s/total*1000:.1f} ms |",
            f"| average CPU time / case | {runtime_s*workers/total:.3f} s |",
        ]
    lines += [f"| output size on disk | {size/1024/1024:.1f} MB |", ""]

    lines += [
        "## Convergence by baseline family", "",
        "| baseline | cases | converged | failed | timed out |",
        "|---|---|---|---|---|",
    ]
    for b, rate, tot, c in sorted(items, key=lambda t: -t[1]):
        lines.append(f"| {b} | {tot} | {c['converged']} ({rate:.1f}%) | "
                     f"{c['failed']} | {c['timed_out']} |")

    lines += [
        "", "## Notes", "",
        "- A few percent of scattered non-convergence is expected and normal "
        "for XFOIL; it is not a defect to chase.",
        "- `MRCHDU: Convergence failed` in a log is a benign *inner* "
        "boundary-layer iteration warning. Convergence truth is whether the "
        "point reached the stored polar.",
        "- Every (airfoil, angle) pair carries an explicit status in "
        "`xfoil_data/status.csv`; nothing is silently dropped.", "",
        "## Plots", "",
        "- `convergence_overview.png` - overall split, per-airfoil rate worst to best",
        "- `coverage_grid.png` - every case, grouped by baseline",
        "- `convergence_by_baseline.png` - per baseline family",
        "- `cl_alpha_samples.png` - lift curves, 10 random airfoils", "",
    ]
    open(out_path, "w").write("\n".join(lines))
    print(f"Saved {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=os.path.join(here, "xfoil_data"))
    p.add_argument("--out_dir", default=os.path.join(here, "xfoil_batch_report"))
    p.add_argument("--runtime_s", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_status(os.path.join(args.data_dir, "status.csv"))
    cp_dir = os.path.join(args.data_dir, "cp")

    plot_overview(rows, os.path.join(args.out_dir, "convergence_overview.png"))
    plot_coverage_grid(rows, os.path.join(args.out_dir, "coverage_grid.png"))
    plot_cl_alpha(cp_dir, os.path.join(args.out_dir, "cl_alpha_samples.png"),
                  seed=args.seed)
    items = plot_by_baseline(rows,
                             os.path.join(args.out_dir, "convergence_by_baseline.png"))
    write_summary(rows, items, os.path.join(args.out_dir, "summary.md"),
                  args.runtime_s, args.workers, cp_dir)


if __name__ == "__main__":
    main()
