# Stage 1 spread over seeds

Seeds run: 42, 43, 44 (3 of 3 planned). Same data, split and code; only the initialisation and batch order change.

| seed | epochs | best epoch | stop reason | minutes |
|---|---|---|---|---|
| 42 | 400 | 376 | max epochs | 24.9 |
| 43 | 281 | 251 | early stop: no val improvement for 30 epochs | 18.0 |
| 44 | 321 | 291 | early stop: no val improvement for 30 epochs | 19.9 |

## val

| metric | seed 42 | seed 43 | seed 44 | mean | std | min to max |
|---|---|---|---|---|---|---|
| Cp RMSE | 0.0329 | 0.0384 | 0.0367 | 0.0360 | 0.0028 | 0.0329 to 0.0384 |
| Cp relative L2 | 0.0371 | 0.0439 | 0.0415 | 0.0408 | 0.0035 | 0.0371 to 0.0439 |
| CL MAE (from Cp) | 0.0110 | 0.0121 | 0.0117 | 0.0116 | 0.0006 | 0.0110 to 0.0121 |

## test

| metric | seed 42 | seed 43 | seed 44 | mean | std | min to max |
|---|---|---|---|---|---|---|
| Cp RMSE | 0.0313 | 0.0372 | 0.0358 | 0.0348 | 0.0031 | 0.0313 to 0.0372 |
| Cp relative L2 | 0.0342 | 0.0405 | 0.0383 | 0.0377 | 0.0032 | 0.0342 to 0.0405 |
| CL MAE (from Cp) | 0.0093 | 0.0107 | 0.0097 | 0.0099 | 0.0007 | 0.0093 to 0.0107 |

## test_unseen_family

| metric | seed 42 | seed 43 | seed 44 | mean | std | min to max |
|---|---|---|---|---|---|---|
| Cp RMSE | 0.1326 | 0.1615 | 0.1442 | 0.1461 | 0.0145 | 0.1326 to 0.1615 |
| Cp relative L2 | 0.1746 | 0.1986 | 0.1835 | 0.1855 | 0.0122 | 0.1746 to 0.1986 |
| CL MAE (from Cp) | 0.0207 | 0.0238 | 0.0334 | 0.0260 | 0.0066 | 0.0207 to 0.0334 |

