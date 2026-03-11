# Multi-run Diagnostics (Manuscript Protocol)

- Seeds: [42, 52, 62, 72, 82]
- Repeats per slice: 40
- Total run rows: 16679

## Key Failure Signals (noisy regime)

- seed=42, Output3 rbf: neg_R2_rate=97.27%, rmse_mean=3.725, rmse_p95=8.159, r2_mean=-52.259
- seed=72, Output3 rbf: neg_R2_rate=96.82%, rmse_mean=3.613, rmse_p95=7.054, r2_mean=-63.427
- seed=62, Output3 rbf: neg_R2_rate=96.14%, rmse_mean=4.191, rmse_p95=8.461, r2_mean=-67.462
- seed=82, Output3 rbf: neg_R2_rate=95.91%, rmse_mean=3.744, rmse_p95=7.848, r2_mean=-57.201
- seed=52, Output3 rbf: neg_R2_rate=92.50%, rmse_mean=3.334, rmse_p95=7.189, r2_mean=-43.999
- seed=62, Output3 cubic: neg_R2_rate=89.76%, rmse_mean=1.953, rmse_p95=3.314, r2_mean=-12.551
- seed=72, Output3 cubic: neg_R2_rate=88.24%, rmse_mean=1.741, rmse_p95=3.035, r2_mean=-11.363
- seed=42, Output3 cubic: neg_R2_rate=85.71%, rmse_mean=1.560, rmse_p95=2.876, r2_mean=-6.895

## Typical Stable Signals (noise-free regime)

- seed=62, Output3 cubic: r2_mean=1.000, rmse_mean=0.007, neg_R2_rate=0.00%
- seed=72, Output3 cubic: r2_mean=1.000, rmse_mean=0.007, neg_R2_rate=0.00%
- seed=42, Output3 cubic: r2_mean=1.000, rmse_mean=0.007, neg_R2_rate=0.00%
- seed=52, Output3 cubic: r2_mean=1.000, rmse_mean=0.007, neg_R2_rate=0.00%
- seed=82, Output3 cubic: r2_mean=1.000, rmse_mean=0.007, neg_R2_rate=0.00%
- seed=82, Output1 cubic: r2_mean=0.993, rmse_mean=0.038, neg_R2_rate=0.00%
- seed=72, Output1 cubic: r2_mean=0.991, rmse_mean=0.045, neg_R2_rate=0.00%
- seed=42, Output1 cubic: r2_mean=0.989, rmse_mean=0.048, neg_R2_rate=0.00%
