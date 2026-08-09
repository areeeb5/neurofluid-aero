"""Batch XFOIL runner over the CST airfoil set (Phase 1, Step 4).

One XFOIL process handles one airfoil's full angle-of-attack sweep: the
boundary-layer solution marches from the previous angle, which is both
faster and better-converging than restarting per angle.

Follows the hard-won reliability rules from Step 3 (see data_spec.md):
  - stdin/stdout are redirected to real FILES, never pipes (piping hangs)
  - every script starts with PLOP / G / blank to disable graphics
  - convergence is judged by whether an angle reached the polar, NOT by
    the absence of "MRCHDU: Convergence failed" warnings (those are
    benign inner boundary-layer iteration messages)

Every (airfoil, alpha) pair gets an explicit status - converged, failed,
or timed_out - recorded in status.csv, which doubles as the resume ledger.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XFOIL_EXE = os.path.join(REPO, "tools", "xfoil", "xfoil.exe")

STATUS_CONVERGED = "converged"
STATUS_FAILED = "failed"
STATUS_TIMED_OUT = "timed_out"

POLAR_HEADER_LINES = 12   # XFOIL PACC polar header
CP_HEADER_LINES = 3       # XFOIL CPWR header
N_PANEL_NODES = 200       # PPAR N -> fixed panel count for every airfoil,
                          # so Cp arrays are a consistent shape (and match
                          # CLAUDE.md's ~200 surface points for Head A)

# XFOIL is single-threaded and CPU-bound; leave two cores for the OS.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 2)


# ---------------------------------------------------------------------------
# XFOIL driving
# ---------------------------------------------------------------------------

def build_command_script(alphas, reynolds: float, mach: float, iters: int) -> str:
    """Build the XFOIL stdin script for one airfoil's full sweep."""
    cmds = [
        "PLOP", "G", "",              # graphics off (headless)
        "LOAD af.dat",
        "PPAR", f"N {N_PANEL_NODES}", "", "",
        "OPER",
        f"VISC {reynolds:.0f}",
        f"MACH {mach}",
        f"ITER {iters}",
        "PACC", "polar.txt", "",      # start polar accumulation, no dump file
    ]
    for i, a in enumerate(alphas):
        cmds += [f"ALFA {a:.4f}", f"CPWR c{i:02d}.txt"]
    cmds += ["PACC", "", "QUIT", ""]
    return "\n".join(cmds)


def parse_polar(path: str):
    """Parse an XFOIL PACC polar -> dict keyed by rounded alpha."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path, errors="replace").read().splitlines()[POLAR_HEADER_LINES:]
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(p) for p in parts[:7]]
        except ValueError:
            continue
        out[round(vals[0], 4)] = {
            "alpha": vals[0], "CL": vals[1], "CD": vals[2], "CDp": vals[3],
            "CM": vals[4], "Top_Xtr": vals[5], "Bot_Xtr": vals[6],
        }
    return out


def parse_cp(path: str):
    """Parse an XFOIL CPWR file -> (x, y, cp), or None if unusable."""
    if not os.path.exists(path):
        return None
    try:
        d = np.loadtxt(path, skiprows=CP_HEADER_LINES)
    except Exception:
        return None
    if d.ndim != 2 or d.shape[0] != N_PANEL_NODES or d.shape[1] < 3:
        return None
    return d[:, 0], d[:, 1], d[:, 2]


def run_one_airfoil(airfoil_id: str, dat_path: str, alphas, args, work_root: str):
    """Run one airfoil's full sweep. Returns (statuses, record, elapsed)."""
    wd = os.path.join(work_root, airfoil_id)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd, exist_ok=True)

    # Short local filename - XFOIL has a limited filename buffer.
    shutil.copy(dat_path, os.path.join(wd, "af.dat"))
    with open(os.path.join(wd, "cmds.inp"), "w") as f:
        f.write(build_command_script(alphas, args.reynolds, args.mach, args.iters))

    timed_out = False
    t0 = time.time()
    try:
        with open(os.path.join(wd, "cmds.inp")) as fin, \
             open(os.path.join(wd, "run.log"), "w") as fout:
            # Real file handles, never PIPE - see data_spec.md Step 3 notes.
            subprocess.run([XFOIL_EXE], stdin=fin, stdout=fout,
                           stderr=subprocess.STDOUT, cwd=wd,
                           timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception:
        pass
    elapsed = time.time() - t0

    polar = parse_polar(os.path.join(wd, "polar.txt"))

    statuses = []
    conv_alphas, cls, cds, cdps, cms, tops, bots, cps = [], [], [], [], [], [], [], []
    x_ref = y_ref = None

    for i, a in enumerate(alphas):
        key = round(a, 4)
        cp_data = parse_cp(os.path.join(wd, f"c{i:02d}.txt"))
        if key in polar and cp_data is not None:
            p = polar[key]
            x, y, cp = cp_data
            if x_ref is None:
                x_ref, y_ref = x, y
            conv_alphas.append(a)
            cls.append(p["CL"]); cds.append(p["CD"]); cdps.append(p["CDp"])
            cms.append(p["CM"]); tops.append(p["Top_Xtr"]); bots.append(p["Bot_Xtr"])
            cps.append(cp)
            statuses.append((airfoil_id, a, STATUS_CONVERGED))
        else:
            statuses.append((airfoil_id, a,
                             STATUS_TIMED_OUT if timed_out else STATUS_FAILED))

    record = None
    if conv_alphas:
        record = {
            "alpha": np.array(conv_alphas), "x": x_ref, "y": y_ref,
            "cp": np.array(cps), "CL": np.array(cls), "CD": np.array(cds),
            "CDp": np.array(cdps), "CM": np.array(cms),
            "Top_Xtr": np.array(tops), "Bot_Xtr": np.array(bots),
        }

    if not args.keep_work:
        shutil.rmtree(wd, ignore_errors=True)
    return statuses, record, elapsed


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def load_index(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_done(status_path: str, n_alphas: int):
    """Airfoils with a full set of recorded alphas are already complete."""
    if not os.path.exists(status_path):
        return set()
    counts = {}
    with open(status_path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["airfoil_id"]] = counts.get(row["airfoil_id"], 0) + 1
    return {a for a, c in counts.items() if c >= n_alphas}


def aggregate_polars(cp_dir: str, out_csv: str) -> int:
    """Rebuild the combined polar table from the per-airfoil .npz files.

    Built from the .npz files rather than appended during the run, so a
    resumed run cannot duplicate rows.
    """
    cols = ["CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    n = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airfoil_id", "baseline", "alpha"] + cols)
        for path in sorted(os.listdir(cp_dir)):
            if not path.endswith(".npz"):
                continue
            aid = path[:-4]
            d = np.load(os.path.join(cp_dir, path), allow_pickle=False)
            baseline = str(d["baseline"]) if "baseline" in d else ""
            for i, a in enumerate(d["alpha"]):
                w.writerow([aid, baseline, f"{a:g}"] +
                           [f"{float(d[c][i]):.6g}" for c in cols])
                n += 1
    return n


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index_csv", default=os.path.join(here, "airfoil_index.csv"))
    p.add_argument("--airfoils_dir", default=os.path.join(here, "airfoils"))
    p.add_argument("--out_dir", default=os.path.join(here, "xfoil_data"))
    p.add_argument("--pilot", type=int, default=0,
                   help="if >0, run this many randomly sampled airfoils")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="parallel XFOIL processes (default: CPU count - 2)")
    p.add_argument("--timeout", type=float, default=240.0,
                   help="seconds for one airfoil's full sweep. Sized from "
                        "measured per-ANGLE timing: slowest single angle "
                        "observed was 3.75s, so a sweep where every angle is "
                        "that slow is ~56s; 240s is ~4.3x margin on that and "
                        "19x the slowest sweep actually observed.")
    p.add_argument("--alpha_min", type=float, default=-4.0)
    p.add_argument("--alpha_max", type=float, default=10.0)
    p.add_argument("--alpha_step", type=float, default=1.0)
    p.add_argument("--reynolds", type=float, default=1e6)
    p.add_argument("--mach", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--keep_work", action="store_true")
    args = p.parse_args()

    if not os.path.exists(XFOIL_EXE):
        print(f"ERROR: xfoil.exe not found at {XFOIL_EXE}", file=sys.stderr)
        return 1

    n_steps = int(round((args.alpha_max - args.alpha_min) / args.alpha_step)) + 1
    alphas = [round(args.alpha_min + i * args.alpha_step, 4) for i in range(n_steps)]

    records = load_index(args.index_csv)
    if args.pilot > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(records), size=min(args.pilot, len(records)),
                         replace=False)
        records = [records[i] for i in sorted(idx)]

    os.makedirs(args.out_dir, exist_ok=True)
    cp_dir = os.path.join(args.out_dir, "cp")
    os.makedirs(cp_dir, exist_ok=True)
    work_root = os.path.join(args.out_dir, "_work")
    os.makedirs(work_root, exist_ok=True)
    status_path = os.path.join(args.out_dir, "status.csv")

    done = load_done(status_path, len(alphas))
    todo = [r for r in records if r["airfoil_id"] not in done]

    print(f"XFOIL batch: {len(records)} airfoils x {len(alphas)} alphas "
          f"({alphas[0]:g}..{alphas[-1]:g} step {args.alpha_step:g})")
    print(f"  Re={args.reynolds:g}  M={args.mach}  ITER={args.iters}  "
          f"panels={N_PANEL_NODES}")
    print(f"  workers={args.workers}  timeout={args.timeout:g}s/airfoil")
    if done:
        print(f"  resuming: {len(done)} airfoils already complete, "
              f"{len(todo)} to run")
    if not todo:
        print("Nothing to do.")
        return 0

    new_file = not os.path.exists(status_path)
    status_f = open(status_path, "a", newline="")
    writer = csv.writer(status_f)
    if new_file:
        writer.writerow(["airfoil_id", "baseline", "alpha", "status"])
        status_f.flush()

    lock = threading.Lock()
    counters = {STATUS_CONVERGED: 0, STATUS_FAILED: 0, STATUS_TIMED_OUT: 0}
    times = []
    completed = 0
    t_start = time.time()

    def work(rec):
        aid = rec["airfoil_id"]
        dat = os.path.join(args.airfoils_dir, rec["filename"])
        return run_one_airfoil(aid, dat, alphas, args, work_root)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            aid = rec["airfoil_id"]
            baseline = rec.get("baseline", "")
            try:
                statuses, record, elapsed = fut.result()
            except Exception as e:  # never lose a case silently
                statuses = [(aid, a, STATUS_FAILED) for a in alphas]
                record, elapsed = None, 0.0
                print(f"  {aid}: worker error: {e}", file=sys.stderr)

            with lock:
                if record is not None:
                    record["baseline"] = baseline
                    np.savez_compressed(os.path.join(cp_dir, f"{aid}.npz"), **record)
                for s in statuses:
                    writer.writerow([s[0], baseline, f"{s[1]:g}", s[2]])
                    counters[s[2]] += 1
                status_f.flush()
                times.append(elapsed)
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    rate = np.mean(times)
                    eta = (len(todo) - completed) * rate / max(args.workers, 1)
                    print(f"  {completed}/{len(todo)} airfoils  "
                          f"avg {rate:.2f}s/airfoil  ETA {eta/60:.1f} min")

    status_f.close()
    wall = time.time() - t_start

    total = sum(counters.values())
    print(f"\nDone in {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  cases: {total}")
    for k in (STATUS_CONVERGED, STATUS_FAILED, STATUS_TIMED_OUT):
        print(f"    {k:10s} {counters[k]:6d}  ({counters[k]/max(total,1)*100:.2f}%)")
    if times:
        t = np.array(times)
        print(f"  per-airfoil sweep: mean {t.mean():.2f}s  median "
              f"{np.median(t):.2f}s  p95 {np.percentile(t,95):.2f}s  "
              f"max {t.max():.2f}s")
        print(f"  per-case (airfoil x alpha): mean "
              f"{t.mean()/len(alphas):.3f}s")

    polars_csv = os.path.join(args.out_dir, "polars.csv")
    n_rows = aggregate_polars(cp_dir, polars_csv)
    print(f"  aggregated {n_rows} converged points -> {polars_csv}")

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
