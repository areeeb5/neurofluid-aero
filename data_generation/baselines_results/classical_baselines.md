# Classical-ML baselines

Inputs: 10 CST coefficients + 2 trailing-edge offsets + angle of attack. Trained on 5063 cases from the `train` split. Seed 42. R2 = 1 is perfect, 0 is no better than predicting the training mean.

## val

| model | R2 CL | R2 CD | R2 CM | MAE CL | MAE CD | MAE CM |
|---|---|---|---|---|---|---|
| mean (no skill) | -0.000 | -0.002 | -0.001 | 0.4667 | 0.0028 | 0.0465 |
| linear | 0.995 | 0.621 | 0.987 | 0.0319 | 0.0017 | 0.0059 |
| random forest | 0.997 | 0.988 | 0.994 | 0.0203 | 0.0003 | 0.0037 |
| gradient boosting | 0.997 | 0.984 | 0.994 | 0.0212 | 0.0003 | 0.0041 |

## test

| model | R2 CL | R2 CD | R2 CM | MAE CL | MAE CD | MAE CM |
|---|---|---|---|---|---|---|
| mean (no skill) | -0.000 | -0.000 | -0.002 | 0.4728 | 0.0028 | 0.0464 |
| linear | 0.995 | 0.598 | 0.990 | 0.0284 | 0.0017 | 0.0053 |
| random forest | 0.998 | 0.976 | 0.993 | 0.0172 | 0.0003 | 0.0045 |
| gradient boosting | 0.998 | 0.970 | 0.994 | 0.0183 | 0.0003 | 0.0039 |

## test_unseen_family

| model | R2 CL | R2 CD | R2 CM | MAE CL | MAE CD | MAE CM |
|---|---|---|---|---|---|---|
| mean (no skill) | -0.137 | -0.010 | -1.056 | 0.4144 | 0.0038 | 0.0374 |
| linear | 0.974 | 0.484 | 0.901 | 0.0446 | 0.0028 | 0.0074 |
| random forest | 0.965 | 0.939 | 0.618 | 0.0727 | 0.0009 | 0.0160 |
| gradient boosting | 0.985 | 0.941 | 0.938 | 0.0437 | 0.0010 | 0.0063 |

## Reading

- On the in-family `test` split the random forest's lowest R2 is **CD** (0.976); Moin et al. 2021 found CD the hardest target.
- Compare `test` against `test_unseen_family`: the gap is how much of the skill is memorising familiar baseline families rather than learning aerodynamics.
- The dataset is 15 tight clusters around real airfoils, so in-family splits are close to interpolation and these baselines score very high there. Whether the neural operator (Stage 1) beats them is an open question; the `test_unseen_family` split is the meaningful bar.
