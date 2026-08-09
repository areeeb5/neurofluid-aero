# Step 4 - batch XFOIL run summary

500 airfoils x 15 angles of attack (-4 deg to 10 deg, 1 deg steps), Re = 1e6, M = 0.15, Ncrit = 9 (default), free transition, 200 panel nodes.

| metric | value |
|---|---|
| total cases | 7500 |
| converged | 7075 (94.33%) |
| failed | 407 (5.43%) |
| timed out | 18 (0.24%) |
| total runtime | 556.2 s (9.3 min) |
| parallel workers | 6 |
| average wall time / case | 74.2 ms |
| average CPU time / case | 0.445 s |
| output size on disk | 10.8 MB |

## Convergence by baseline family

| baseline | cases | converged | failed | timed out |
|---|---|---|---|---|
| naca23012 | 495 | 490 (99.0%) | 5 | 0 |
| naca63-412 | 495 | 487 (98.4%) | 8 | 0 |
| naca4412 | 510 | 500 (98.0%) | 10 | 0 |
| clarky | 495 | 483 (97.6%) | 12 | 0 |
| e387 | 495 | 481 (97.2%) | 14 | 0 |
| s1223 | 495 | 480 (97.0%) | 15 | 0 |
| naca0012 | 510 | 494 (96.9%) | 16 | 0 |
| rae2822 | 495 | 479 (96.8%) | 16 | 0 |
| sd7062 | 495 | 479 (96.8%) | 16 | 0 |
| naca2412 | 510 | 492 (96.5%) | 18 | 0 |
| naca4415 | 495 | 474 (95.8%) | 21 | 0 |
| naca0018 | 510 | 485 (95.1%) | 25 | 0 |
| goe387 | 495 | 443 (89.5%) | 52 | 0 |
| naca6409 | 495 | 438 (88.5%) | 57 | 0 |
| naca0006 | 510 | 370 (72.5%) | 122 | 18 |

## Notes

- A few percent of scattered non-convergence is expected and normal for XFOIL; it is not a defect to chase.
- `MRCHDU: Convergence failed` in a log is a benign *inner* boundary-layer iteration warning. Convergence truth is whether the point reached the stored polar.
- Every (airfoil, angle) pair carries an explicit status in `xfoil_data/status.csv`; nothing is silently dropped.

## Plots

- `convergence_overview.png` - overall split, per-airfoil rate worst to best
- `coverage_grid.png` - every case, grouped by baseline
- `convergence_by_baseline.png` - per baseline family
- `cl_alpha_samples.png` - lift curves, 10 random airfoils
