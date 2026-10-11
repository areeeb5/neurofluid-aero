# Stage 1 error by baseline family (seed 42)

Mean relative Cp error (L2) per airfoil, averaged over a family. `train` airfoils were fitted by the model; `held-out` = val + test + unseen-family airfoils.

| baseline | mean thickness | airfoils | train error | held-out error |
|---|---|---|---|---|
| naca0006 | 6.0% | 34 | 4.5% | 5.4% |
| naca6409 | 8.8% | 32 | n/a (held out) | 17.8% |
| e387 | 9.2% | 33 | n/a (held out) | 17.4% |
| naca4412 | 11.9% | 34 | 1.8% | 2.3% |
| clarky | 11.9% | 33 | 1.9% | 2.6% |
| naca2412 | 11.9% | 34 | 1.9% | 3.2% |
| rae2822 | 12.0% | 33 | 2.4% | 5.0% |
| naca0012 | 12.0% | 34 | 2.0% | 3.7% |
| naca23012 | 12.0% | 33 | 2.1% | 3.2% |
| naca63-412 | 12.1% | 33 | 2.2% | 4.2% |
| s1223 | 12.2% | 33 | 1.5% | 5.1% |
| sd7062 | 13.8% | 33 | 1.8% | 2.5% |
| naca4415 | 15.0% | 33 | 2.0% | 2.8% |
| goe387 | 15.0% | 33 | 2.0% | 4.1% |
| naca0018 | 18.1% | 34 | 1.6% | 2.8% |

Correlation of airfoil error with thickness over the 78 val+test airfoils: **-0.35** (negative = thinner airfoils have larger error).

## Reading

- The two held-out families (`naca6409`, `e387`) have about 17 to 18% error against 2 to 5% for families the model trained on: generalising to an unseen family is the main gap.
- `naca0006`, the thinnest family, has roughly twice the error of the others even on its TRAINING airfoils. This is consistent with, but does not prove, thin sections being resolved coarsely by the 64x64 grid (their mask is only 2 cells thick).
- Separately, a single-case diagnostic (`overfit_single_sample.py --hardest`) shows the network can memorise even the sharpest suction spike in the data (Cp down to -8.84) to 0.3% error, so the spike errors on unseen cases are not a capacity limit of the network.
