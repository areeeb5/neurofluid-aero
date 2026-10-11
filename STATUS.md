# STATUS

**Updated:** 2026-10-11 06:32 (overnight run finished; stopped well before the 07:45 hard stop)
**Current state:** All approved overnight steps done and committed on local branch overnight-2026-10-11 (NOT pushed; main = bdab826, in sync with the areeeb5 repo). Existing data verified unchanged (checksums). Done: 64x64 grid format (x -0.25..1.25, y -0.75..0.75), XFOIL QA (0 flags), split with 2 held-out families, classical baselines, Stage 1 (FNO + Head A) with overfit gate passed, trained on 3 seeds.
**Last result:** Stage 1, 3-seed mean: test Cp RMSE 0.0348 (3.8% rel), held-out families 0.146 (18.6%). CL from predicted Cp beats the baselines 1.7x in-family and 1.7x to 2.8x on unseen families. Seed 42 was the best run; seeds differ by about 10%.
**Next step:** Areeb reads overnight/MORNING_REPORT.md. Then: tell the ANSYS session the grid domain, review/merge/push the branch, one validated ANSYS case on the new grid.
**Blocked / waiting on Areeb:** (1) ANSYS Meshing session has NOT been told the grid domain (export must be x -0.25..1.25, y -0.75..0.75, 64x64 cell centres, [iy, ix]). (2) Review, merge and push of the overnight branch. (3) Active selection is a draft only; a real one needs Stage 1 retrained on all families. (4) START_PROMPT.md still untracked, left alone on purpose.
**Recommended model/effort:** Sonnet / low to read the report; Opus / medium to plan ANSYS case 1 on the new grid.
**Also:** Plain-English guide (updated): https://claude.ai/code/artifact/1b697394-9276-4145-a330-f38eb92c9631
