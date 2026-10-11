# Stage 1 results (Head A, Cp on XFOIL data)

Seed 44. 321 epochs in 19.9 min (early stop: no val improvement for 30 epochs); best validation epoch 291. 2,384,865 parameters.

| split | cases | Cp RMSE | Cp relative L2 | CL RMSE (from Cp) | CL R2 | Cp RMSE of a mean-curve guess |
|---|---|---|---|---|---|---|
| train | 5063 | 0.0187 | 0.0244 | 0.0098 | 0.9997 | nan |
| val | 550 | 0.0367 | 0.0415 | 0.0181 | 0.9989 | 0.6621 |
| test | 543 | 0.0358 | 0.0383 | 0.0135 | 0.9994 | 0.6660 |
| test_unseen_family | 919 | 0.1442 | 0.1835 | 0.0450 | 0.9900 | 0.6836 |
