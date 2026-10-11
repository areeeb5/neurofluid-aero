# Stage 1 results (Head A, Cp on XFOIL data)

Seed 42. 400 epochs in 24.9 min (max epochs); best validation epoch 376. 2,384,865 parameters.

| split | cases | Cp RMSE | Cp relative L2 | CL RMSE (from Cp) | CL R2 | Cp RMSE of a mean-curve guess |
|---|---|---|---|---|---|---|
| train | 5063 | 0.0154 | 0.0209 | 0.0087 | 0.9998 | nan |
| val | 550 | 0.0329 | 0.0371 | 0.0165 | 0.9991 | 0.6621 |
| test | 543 | 0.0313 | 0.0342 | 0.0132 | 0.9995 | 0.6660 |
| test_unseen_family | 919 | 0.1326 | 0.1746 | 0.0274 | 0.9963 | 0.6836 |
