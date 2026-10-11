# XFOIL data QA

7075 converged cases checked. Flags are advisory: nothing was deleted or changed. Full list: `xfoil_data/qa_flags.csv`.

| metric | value |
|---|---|
| cases with a HARD flag | 0 (0.00%) |
| cases with only REVIEW flags | 0 (0.00%) |
| clean cases | 7075 (100.00%) |

## Flags by check

| severity | check | count |
|---|---|---|

## Does the Cp curve reproduce the reported CL?

CL from integrating the stored Cp curve minus the polar's CL, over every converged case:

- mean -0.0000, median -0.0000, std 0.0000
- 95% of cases within 0.0000, 99% within 0.0001, max 0.0001
- This is a file-integrity check, not a physics validation: XFOIL computes its CL from this same Cp distribution, so agreement shows the saved Cp curves are complete, correctly ordered and matched to the right angle (no mix-ups), not that the flow solution is physically right.

## HARD flags by baseline

None.
