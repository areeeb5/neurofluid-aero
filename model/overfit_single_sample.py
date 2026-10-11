"""Overfit-a-single-sample gate (required by CLAUDE.md before trusting a training loop).

Trains Stage1Net on ONE case and demands it memorise it:
  PASS  relative L2 error ||pred - target|| / ||target|| < 0.02 within --steps.
A control run with lr = 0 must FAIL, proving the gate can fail.
Exit code 0 only if the real run passes AND the control fails.
Writes overnight/overfit_gate.json and overnight/overfit_gate.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fno  # noqa: E402
from data_stage1 import Stage1Data  # noqa: E402

ROOT = os.path.dirname(HERE)
GATE = 0.02


def run(data: Stage1Data, case: int, lr: float, steps: int, seed: int, device):
    torch.manual_seed(seed)
    net = fno.Stage1Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g_in, sxy, cp, _, _ = data.batch(torch.tensor([case], device=device))
    denom = cp.norm()
    hist = []
    for step in range(steps + 1):
        pred = net(g_in, sxy)
        rel = float((pred - cp).norm() / denom)
        hist.append(rel)
        if step == steps:
            break
        loss = torch.mean((pred - cp) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return hist, net(g_in, sxy).detach().cpu().numpy()[0], cp.cpu().numpy()[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Stage1Data(device)
    rng = np.random.default_rng(args.seed)
    tr = data.indices("train").cpu().numpy()
    case = int(rng.choice(tr))
    ai = int(data.case_ai[case])
    print(f"device {device}; case {case}: {data.ids[ai]} ({data.baseline[data.ids[ai]]}), "
          f"alpha {float(data.case_alpha[case]):g} deg, "
          f"Cp range [{float(data.case_cp[case].min()):.2f}, {float(data.case_cp[case].max()):.2f}]")

    t0 = time.time()
    hist, pred, tgt = run(data, case, args.lr, args.steps, args.seed, device)
    secs = time.time() - t0
    final = hist[-1]
    first_below = next((i for i, h in enumerate(hist) if h < GATE), None)
    passed = final < GATE
    print(f"real run : rel L2 start {hist[0]:.3f} -> final {final:.4f} "
          f"(first below {GATE} at step {first_below}); {secs:.0f}s -> {'PASS' if passed else 'FAIL'}")

    hist0, _, _ = run(data, case, 0.0, 200, args.seed, device)
    control_failed = hist0[-1] >= GATE
    print(f"control  : lr = 0, rel L2 stays {hist0[-1]:.3f} -> "
          f"{'FAILS the gate as it should' if control_failed else 'WRONGLY PASSES'}")

    ok = passed and control_failed
    out = {"case": case, "airfoil": data.ids[ai], "alpha": float(data.case_alpha[case]),
           "steps": args.steps, "lr": args.lr, "seed": args.seed, "gate_rel_l2": GATE,
           "start_rel_l2": hist[0], "final_rel_l2": final, "first_step_below_gate": first_below,
           "control_final_rel_l2": hist0[-1], "gate_passed": bool(ok)}
    os.makedirs(os.path.join(ROOT, "overnight"), exist_ok=True)
    with open(os.path.join(ROOT, "overnight", "overfit_gate.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist, label="real run")
    ax[0].semilogy(hist0, label="control (lr = 0)")
    ax[0].axhline(GATE, color="k", ls="--", lw=0.8, label=f"gate {GATE}")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("relative L2 error"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(tgt, label="XFOIL Cp"); ax[1].plot(pred, "--", label="model Cp")
    ax[1].invert_yaxis(); ax[1].set_xlabel("surface node (TE -> LE -> TE)"); ax[1].set_ylabel("Cp")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Overfit-a-single-sample gate: {data.ids[ai]}, alpha {float(data.case_alpha[case]):g} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "overnight", "overfit_gate.png"), dpi=130)
    print("GATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
