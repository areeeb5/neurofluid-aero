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
| 05:22 | 5 | Stage 1 model built (`model/`): FNO trunk 4 layers x width 32 x 12 modes + Head A, 2,384,865 params. 9/9 model tests pass (translation equivariance, grid-sample alignment vs numpy incl. a known-bad, gradients, CL-from-Cp parity). **Overfit-a-single-sample gate PASSED**: rel L2 1.07 -> 0.0046, crossed the 0.02 gate at step 372; the lr=0 control correctly failed |
| 05:24 | 6 | Stage 1 training launched (seed 42, cap 45 min, early stopping on val). First launch attempt failed on shell quoting, relaunched |
| 05:49 | 6 | Stage 1 training finished (seed 42): 400 epochs in 24.9 min (hit the epoch cap, not early stopping; best val epoch 376, still creeping down). Cp RMSE: train 0.0154, val 0.0329, test 0.0313, unseen-family 0.1326 (guessing the mean curve gives about 0.66). CL from predicted Cp: test MAE 0.0093 (random forest 0.0172), unseen-family MAE 0.0207 (gradient boosting 0.0437, random forest 0.0727). Worst errors are the leading-edge suction spike at high alpha |
| 05:51 | 7 | Active-selection DRAFT written. Top ranks are the two held-out families, by construction, so I added a val+test-only view and caveats |
| 05:50 | extra | Not in the approved list: re-running Stage 1 with seeds 43 and 44 (same code, only compute) to get a spread, since one seed is the main caveat. Outputs are tagged so seed 42 is untouched |
| 05:56 | correction | I had written (in commit 3d41ba6's message and an earlier draft of data_spec) that the worst Cp errors are "where the 64x64 grid resolves the nose coarsely". That was an untested guess. A diagnostic (`overfit_single_sample.py --hardest`) shows the network memorises the sharpest-spike case (Cp to -8.84) to 0.3%, so spike errors on unseen cases are not a capacity limit. Per-family errors (`stage1_error_by_family.md`): unseen families 17 to 18% vs 2 to 5% in-family; thin NACA 0006 about 2x the others even on training airfoils (consistent with coarse thin sections, not proof). data_spec corrected; the commit message stays as is (history not rewritten) |
| 06:30 | extra | Seeds 43 and 44 finished (early-stopped at 281 and 321 epochs). 3-seed mean: test Cp RMSE 0.0348 +/- 0.0031 (seed 42's 0.0313 was the best), unseen-family 0.1461 +/- 0.0145. CL MAE from Cp: test 0.0099 vs random forest 0.0172 (1.7x better), unseen 0.0260 vs gradient boosting 0.0437 / random forest 0.0727 (1.7x to 2.8x). Every document had said "about 2x" from seed 42 alone; all corrected to the 3-seed numbers |
| 06:31 | 8 | Wrap-up: existing-data checksums identical before and after; MORNING_REPORT.md, data_spec.md, CLAUDE.md status, STATUS.md and the project guide updated; no processes left running. Final commit next |
| 05:19 | env | Python on PATH is now 3.14.8 (not the 3.12 installed earlier). PyTorch 2.11.0+cu128 (cp314) downloaded, installing |
