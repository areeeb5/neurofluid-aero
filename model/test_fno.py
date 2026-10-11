"""Unit tests for the Stage 1 network and data plumbing. Run: python model/test_fno.py

Known-property tests (not just "it runs"):
  - SpectralConv2d is translation-equivariant (a defining FNO property)
  - sample_latent agrees with the numpy reference grid_encoding.sample_field,
    which is what guarantees Head A reads the cell it thinks it reads
  - gradients reach every parameter
  - batched torch CL-from-Cp equals the numpy QA implementation on real data
Exits nonzero on any failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_generation"))

import fno  # noqa: E402
import grid_encoding as ge  # noqa: E402
from data_stage1 import Stage1Data, cl_from_cp_torch  # noqa: E402
from qa_xfoil import cl_from_cp  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}\n")

    net = fno.Stage1Net().to(dev)
    n_par = fno.count_parameters(net)
    print(f"  Stage1Net parameters: {n_par:,}")

    # 1. shapes
    gin = torch.randn(4, 6, 64, 64, device=dev)
    xy = torch.rand(4, 200, 2, device=dev) * 1.0
    out = net(gin, xy)
    check("forward shape (4, 6, 64, 64) + (4, 200, 2) -> (4, 200)", tuple(out.shape) == (4, 200))
    check("trunk latent shape (4, 32, 64, 64)", tuple(net.trunk(gin).shape) == (4, 32, 64, 64))

    # 2. translation equivariance of the spectral layer (circular shift)
    sc = fno.SpectralConv2d(8, 8, 12, 12).to(dev)
    x = torch.randn(2, 8, 64, 64, device=dev)
    a = sc(torch.roll(x, shifts=(3, 5), dims=(2, 3)))
    b = torch.roll(sc(x), shifts=(3, 5), dims=(2, 3))
    err = float((a - b).abs().max())
    check("SpectralConv2d is translation-equivariant", err < 1e-4, f"max err {err:.2e}")

    # 3. sample_latent == numpy reference sampler (incl. clamped points outside the domain)
    rng = np.random.default_rng(0)
    field = rng.standard_normal((64, 64)).astype(np.float32)
    px = rng.uniform(-0.4, 1.4, 300)
    py = rng.uniform(-0.9, 0.9, 300)  # some points fall outside the domain on purpose
    ref = ge.sample_field(field, px, py)
    lat = torch.from_numpy(field).view(1, 1, 64, 64).to(dev)
    pts = torch.from_numpy(np.stack([px, py], -1)).float().view(1, -1, 2).to(dev)
    got = fno.sample_latent(lat, pts)[0, :, 0].cpu().numpy()
    err = float(np.abs(got - ref).max())
    check("sample_latent matches numpy reference sampler", err < 1e-5, f"max err {err:.2e}")

    # 3b. known-bad: a transposed field must NOT match (the test can fail)
    bad = fno.sample_latent(lat.transpose(2, 3).contiguous(), pts)[0, :, 0].cpu().numpy()
    check("known-bad: transposed field is detected", float(np.abs(bad - ref).max()) > 1e-2)

    # 4. gradient reaches every parameter
    net.zero_grad()
    net(gin, xy).pow(2).mean().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    check("gradients are finite for every parameter", not missing, f"bad: {missing[:3]}")

    # 5. torch CL-from-Cp == numpy QA implementation on real cases
    data = Stage1Data(dev)
    idx = data.indices("train")[:64]
    g_in, sxy, cp, alpha, cl = data.batch(idx)
    cl_t = cl_from_cp_torch(sxy, cp, alpha).cpu().numpy()
    cl_n = np.array([cl_from_cp(sxy[i, :, 0].cpu().numpy(), sxy[i, :, 1].cpu().numpy(),
                                cp[i].cpu().numpy(), float(alpha[i]))[0] for i in range(len(idx))])
    err = float(np.abs(cl_t - cl_n).max())
    check("torch CL-from-Cp equals numpy implementation", err < 1e-4, f"max err {err:.2e}")
    err = float(np.abs(cl_t - cl.cpu().numpy()).max())
    check("CL from stored Cp reproduces stored CL (64 real cases)", err < 2e-3, f"max err {err:.2e}")

    # 6. assembled input channels are what the spec says
    ok = (g_in.shape[1:] == (6, 64, 64)
          and torch.allclose(g_in[:, 4, 0, 0], alpha / 10.0)
          and torch.allclose(g_in[:, 5], torch.ones_like(g_in[:, 5])))
    check("batch input: alpha channel = alpha/10, Re channel = 1.0", bool(ok))
    print(f"\n  dataset: {data.n_cases} converged cases; "
          + ", ".join(f"{s} {len(data.indices(s))}" for s in ["train", "val", "test", "test_unseen_family"]))

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("All Stage 1 model tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
