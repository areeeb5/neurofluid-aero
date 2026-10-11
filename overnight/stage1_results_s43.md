# Stage 1 results (Head A, Cp on XFOIL data)

Seed 43. 281 epochs in 18.0 min (early stop: no val improvement for 30 epochs); best validation epoch 251. 2,384,865 parameters.

| split | cases | Cp RMSE | Cp relative L2 | CL RMSE (from Cp) | CL R2 | Cp RMSE of a mean-curve guess |
|---|---|---|---|---|---|---|
| train | 5063 | 0.0201 | 0.0274 | 0.0119 | 0.9995 | nan |
| val | 550 | 0.0384 | 0.0439 | 0.0176 | 0.9990 | 0.6621 |
| test | 543 | 0.0372 | 0.0405 | 0.0143 | 0.9994 | 0.6660 |
| test_unseen_family | 919 | 0.1615 | 0.1986 | 0.0309 | 0.9953 | 0.6836 |
