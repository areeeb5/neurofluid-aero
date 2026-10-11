# DRAFT active selection (not sent to ANSYS)

Top 20 airfoils by mean relative Cp error of the Stage 1 model, at most 3 per baseline family, airfoils with at least 8 converged angles. `worst_alpha` is the suggested ANSYS angle (largest error).

Caveats:
- Train-split errors are optimistic (the model has fitted those airfoils).
- The top ranks are airfoils from the two families held out of training (`e387`, `naca6409`). They rank high BY CONSTRUCTION: the model never saw them. A real selection should first retrain Stage 1 on all families. The third view (val + test airfoils from families the model DID train on) is the fairer picture of ordinary generalisation error.
- A large error at high alpha on a thin airfoil may be XFOIL near stall rather than a model failure; check before spending ANSYS time.
- Check that each suggested angle is a sensible ANSYS operating point before using it.

## all airfoils

| rank | airfoil | baseline | split | mean Cp rel. L2 | worst alpha | worst-case rel. L2 |
|---|---|---|---|---|---|---|
| 1 | airfoil_0288 | e387 | test_unseen_family | 0.244 | -3 | 0.351 |
| 2 | airfoil_0177 | naca6409 | test_unseen_family | 0.230 | -1 | 0.312 |
| 3 | airfoil_0185 | naca6409 | test_unseen_family | 0.222 | -2 | 0.344 |
| 4 | airfoil_0195 | naca6409 | test_unseen_family | 0.213 | -1 | 0.299 |
| 5 | airfoil_0297 | e387 | test_unseen_family | 0.211 | -3 | 0.348 |
| 6 | airfoil_0272 | e387 | test_unseen_family | 0.195 | -3 | 0.298 |
| 7 | airfoil_0319 | s1223 | val | 0.091 | -2 | 0.138 |
| 8 | airfoil_0315 | s1223 | test | 0.076 | -4 | 0.102 |
| 9 | airfoil_0024 | naca0006 | val | 0.074 | 1 | 0.109 |
| 10 | airfoil_0016 | naca0006 | test | 0.074 | -3 | 0.128 |
| 11 | airfoil_0435 | rae2822 | test | 0.073 | 9 | 0.163 |
| 12 | airfoil_0007 | naca0006 | train | 0.071 | 2 | 0.106 |
| 13 | airfoil_0415 | naca63-412 | val | 0.065 | -4 | 0.093 |
| 14 | airfoil_0408 | naca63-412 | val | 0.065 | -2 | 0.092 |
| 15 | airfoil_0346 | goe387 | test | 0.061 | -4 | 0.103 |
| 16 | airfoil_0451 | rae2822 | test | 0.056 | 5 | 0.063 |
| 17 | airfoil_0066 | naca0012 | val | 0.051 | -4 | 0.064 |
| 18 | airfoil_0055 | naca0012 | val | 0.050 | 4 | 0.066 |
| 19 | airfoil_0462 | rae2822 | test | 0.049 | -4 | 0.065 |
| 20 | airfoil_0360 | goe387 | val | 0.045 | -2 | 0.062 |

Families in this list: e387 x3, naca6409 x3, naca0006 x3, rae2822 x3, s1223 x2, naca63-412 x2, goe387 x2, naca0012 x2

## held-out airfoils (val, test, unseen family)

| rank | airfoil | baseline | split | mean Cp rel. L2 | worst alpha | worst-case rel. L2 |
|---|---|---|---|---|---|---|
| 1 | airfoil_0288 | e387 | test_unseen_family | 0.244 | -3 | 0.351 |
| 2 | airfoil_0177 | naca6409 | test_unseen_family | 0.230 | -1 | 0.312 |
| 3 | airfoil_0185 | naca6409 | test_unseen_family | 0.222 | -2 | 0.344 |
| 4 | airfoil_0195 | naca6409 | test_unseen_family | 0.213 | -1 | 0.299 |
| 5 | airfoil_0297 | e387 | test_unseen_family | 0.211 | -3 | 0.348 |
| 6 | airfoil_0272 | e387 | test_unseen_family | 0.195 | -3 | 0.298 |
| 7 | airfoil_0319 | s1223 | val | 0.091 | -2 | 0.138 |
| 8 | airfoil_0315 | s1223 | test | 0.076 | -4 | 0.102 |
| 9 | airfoil_0024 | naca0006 | val | 0.074 | 1 | 0.109 |
| 10 | airfoil_0016 | naca0006 | test | 0.074 | -3 | 0.128 |
| 11 | airfoil_0435 | rae2822 | test | 0.073 | 9 | 0.163 |
| 12 | airfoil_0415 | naca63-412 | val | 0.065 | -4 | 0.093 |
| 13 | airfoil_0408 | naca63-412 | val | 0.065 | -2 | 0.092 |
| 14 | airfoil_0346 | goe387 | test | 0.061 | -4 | 0.103 |
| 15 | airfoil_0451 | rae2822 | test | 0.056 | 5 | 0.063 |
| 16 | airfoil_0066 | naca0012 | val | 0.051 | -4 | 0.064 |
| 17 | airfoil_0055 | naca0012 | val | 0.050 | 4 | 0.066 |
| 18 | airfoil_0462 | rae2822 | test | 0.049 | -4 | 0.065 |
| 19 | airfoil_0026 | naca0006 | val | 0.046 | 3 | 0.055 |
| 20 | airfoil_0360 | goe387 | val | 0.045 | -2 | 0.062 |

Families in this list: e387 x3, naca6409 x3, naca0006 x3, rae2822 x3, s1223 x2, naca63-412 x2, goe387 x2, naca0012 x2

## val + test airfoils only (families the model trained on)

| rank | airfoil | baseline | split | mean Cp rel. L2 | worst alpha | worst-case rel. L2 |
|---|---|---|---|---|---|---|
| 1 | airfoil_0319 | s1223 | val | 0.091 | -2 | 0.138 |
| 2 | airfoil_0315 | s1223 | test | 0.076 | -4 | 0.102 |
| 3 | airfoil_0024 | naca0006 | val | 0.074 | 1 | 0.109 |
| 4 | airfoil_0016 | naca0006 | test | 0.074 | -3 | 0.128 |
| 5 | airfoil_0435 | rae2822 | test | 0.073 | 9 | 0.163 |
| 6 | airfoil_0415 | naca63-412 | val | 0.065 | -4 | 0.093 |
| 7 | airfoil_0408 | naca63-412 | val | 0.065 | -2 | 0.092 |
| 8 | airfoil_0346 | goe387 | test | 0.061 | -4 | 0.103 |
| 9 | airfoil_0451 | rae2822 | test | 0.056 | 5 | 0.063 |
| 10 | airfoil_0066 | naca0012 | val | 0.051 | -4 | 0.064 |
| 11 | airfoil_0055 | naca0012 | val | 0.050 | 4 | 0.066 |
| 12 | airfoil_0462 | rae2822 | test | 0.049 | -4 | 0.065 |
| 13 | airfoil_0026 | naca0006 | val | 0.046 | 3 | 0.055 |
| 14 | airfoil_0360 | goe387 | val | 0.045 | -2 | 0.062 |
| 15 | airfoil_0343 | goe387 | val | 0.043 | -4 | 0.086 |
| 16 | airfoil_0335 | s1223 | val | 0.041 | -4 | 0.058 |
| 17 | airfoil_0227 | naca23012 | val | 0.040 | -3 | 0.062 |
| 18 | airfoil_0122 | naca2412 | val | 0.039 | 2 | 0.064 |
| 19 | airfoil_0233 | naca23012 | val | 0.038 | -4 | 0.063 |
| 20 | airfoil_0076 | naca0018 | test | 0.038 | 8 | 0.047 |

Families in this list: s1223 x3, naca0006 x3, rae2822 x3, goe387 x3, naca63-412 x2, naca0012 x2, naca23012 x2, naca2412 x1, naca0018 x1

