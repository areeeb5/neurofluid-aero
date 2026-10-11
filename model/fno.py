"""Stage 1 network: FNO trunk + Head A (surface pressure coefficient).

Architecture is the agreed design in CLAUDE.md:
  - input: 6 channels on a 64x64 grid [sdf, mask, x, y, alpha, Re]
  - shared trunk: 4 Fourier layers, each SpectralConv2d + 1x1-conv skip + GELU,
    width 32, ~12 modes to start (swept in Phase 3)
  - Head A: bilinearly samples the trunk's latent field at the ~200 surface
    points, then a small MLP predicts Cp at each point

The only addition is a 1x1 "lifting" conv (6 -> width) in front of the Fourier
layers, which any FNO needs to reach the working width.

Grid convention (see data_spec.md): tensors are (B, C, H=iy, W=ix), iy
increasing with y, cell-centre sampling. Surface points are given in physical
chord units and mapped with grid_sample(align_corners=False).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data_generation"))
import grid_encoding as ge  # noqa: E402

N_INPUT_CHANNELS = 6


class SpectralConv2d(nn.Module):
    """Learned multiplication of the lowest Fourier modes (2D, real input).

    Keeps modes1 x modes2 modes in each of the two retained corners of the
    rfft2 spectrum. Weights are stored as real tensors with a trailing (re, im)
    dimension so every optimiser handles them."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes1, self.modes2 = in_ch, out_ch, modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_low = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w_high = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    @staticmethod
    def _mul(x_ft: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x_ft, torch.view_as_complex(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m1, m2 = self.modes1, self.modes2
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_ch, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w_low)
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w_high)
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FourierLayer(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.skip = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.skip(x))


class FNOTrunk(nn.Module):
    def __init__(self, in_ch: int = N_INPUT_CHANNELS, width: int = 32, modes: int = 12,
                 n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.layers = nn.ModuleList([FourierLayer(width, modes) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        return x  # latent field (B, width, 64, 64)


def sample_latent(latent: torch.Tensor, surf_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample the latent field at physical (x, y) points.

    latent: (B, C, H, W); surf_xy: (B, N, 2) in chord units. Returns (B, N, C).
    Matches grid_encoding.sample_field (border-clamped, cell-centre aligned)."""
    u = (surf_xy[..., 0] - ge.X_MIN) / (ge.X_MAX - ge.X_MIN) * 2.0 - 1.0
    v = (surf_xy[..., 1] - ge.Y_MIN) / (ge.Y_MAX - ge.Y_MIN) * 2.0 - 1.0
    grid = torch.stack([u, v], dim=-1).unsqueeze(1)  # (B, 1, N, 2)
    out = F.grid_sample(latent, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)         # (B, C, 1, N)
    return out.squeeze(2).transpose(1, 2)


class HeadA(nn.Module):
    """Latent field sampled at surface points + their coordinates -> Cp."""

    def __init__(self, width: int = 32, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor, surf_xy: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([sample_latent(latent, surf_xy), surf_xy], dim=-1)
        return self.mlp(feats).squeeze(-1)  # (B, N)


class Stage1Net(nn.Module):
    def __init__(self, width: int = 32, modes: int = 12, n_layers: int = 4, hidden: int = 128):
        super().__init__()
        self.trunk = FNOTrunk(N_INPUT_CHANNELS, width, modes, n_layers)
        self.head_a = HeadA(width, hidden)

    def forward(self, grid_in: torch.Tensor, surf_xy: torch.Tensor) -> torch.Tensor:
        return self.head_a(self.trunk(grid_in), surf_xy)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
