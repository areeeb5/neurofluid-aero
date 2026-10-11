# Overnight run log, 2026-10-11

Approved by Areeb. Window 05:12 to 07:45 (PKT), wrap-up by 08:00. Branch: `overnight-2026-10-11`, local commits only, no push.

Locked decisions: grid domain Tight (x -0.25..1.25, y -0.75..0.75); PyTorch (CUDA) + scikit-learn downloads approved; extras off.

| time | step | note |
| --- | --- | --- |
| 05:12 | 0 | Branch created, existing-data checksums saved, keep-awake held, scikit-learn + PyTorch installs started in background |
| 05:12 to 05:15 | 1 | Grid encoding done: `grid_encoding.py`, 17/17 unit tests pass, 500 grids built (6 MB), surface nodes within 0.52 cell of SDF=0, spec written. Finding: 34 NACA 0006 airfoils have a 2-cell-thick mask. Commit 824c6af |
| 05:17 | 2 | XFOIL QA done: 0 HARD and 0 REVIEW flags over 7,075 converged cases. Cp integrates back to the reported CL to 1e-4 (a file-integrity check, not physics: XFOIL derives CL from the same Cp). 7 known-good/bad self-tests pass (6 injected faults all caught). Note: an earlier draft of this log carried an invented timestamp for step 1, now corrected |
| 05:19 | 3 | Split done (`make_splits.py`, seed 42): 356 train / 39 val / 39 test airfoils + 66 in 2 held-out families (`e387`, `naca6409`). 8/8 checks pass incl. a leak detector |
| 05:19 | 4 | Classical baselines done: RF `test` R2 CL 0.998 / CD 0.976 / CM 0.993; on `test_unseen_family` 0.965 / 0.939 / 0.618. CD hardest in-family (matches Moin 2021) |
| 05:19 | env | Python on PATH is now 3.14.8 (not the 3.12 installed earlier). PyTorch 2.11.0+cu128 (cp314) downloaded, installing |
