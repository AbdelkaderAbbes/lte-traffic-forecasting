## Data audit

- Raw rows: 1800920, cells: 65 kept of 184, span: 86.5 days (2025-02-07 19:45:00 to 2025-05-05 07:15:00)
- Exact duplicate rows removed: 0; same-key rows merged: 1078375
- RB utilisation NaN share: 0.6426; of which rows with DL volume reported as 0 (idle, set to 0 %): 0.0403
- Idle slots (RB = 0) in the grid: 0.0626; true gaps: 0.2941
- Share of RB values above 85 %: 0.0
- Max RB utilisation in the data: 74.5 %

## RB utilisation: MAE (%-points), mean over test folds

| horizon | xgb   | persistence | seasonal_naive_1d | seasonal_naive_7d | seasonal_mean_7d |
| ------- | ----- | ----------- | ----------------- | ----------------- | ---------------- |
| 1       | 3.784 | 4.913       | 5.913             | 6.228             | 5.079            |
| 2       | 4.069 | 6.203       | 5.911             | 6.226             | 5.078            |
| 3       | 4.223 | 7.406       | 5.908             | 6.225             | 5.077            |
| 4       | 4.321 | 8.557       | 5.908             | 6.224             | 5.076            |
| 5       | 4.406 | 9.590       | 5.910             | 6.226             | 5.076            |
| 6       | 4.457 | 10.429      | 5.913             | 6.230             | 5.076            |

## Model vs strongest baseline (pooled test rows; 95 % block-bootstrap CI of MAE difference, model - baseline)

| horizon | best_baseline    | mae_model | mae_baseline | diff   | diff_lo | diff_hi | skill_pct |
| ------- | ---------------- | --------- | ------------ | ------ | ------- | ------- | --------- |
| 1       | persistence      | 3.785     | 4.914        | -1.130 | -1.167  | -1.093  | 22.985    |
| 2       | seasonal_mean_7d | 4.070     | 5.081        | -1.010 | -1.089  | -0.926  | 19.888    |
| 3       | seasonal_mean_7d | 4.225     | 5.080        | -0.855 | -0.930  | -0.779  | 16.825    |
| 4       | seasonal_mean_7d | 4.323     | 5.079        | -0.756 | -0.827  | -0.684  | 14.879    |
| 5       | seasonal_mean_7d | 4.409     | 5.078        | -0.670 | -0.738  | -0.596  | 13.188    |
| 6       | seasonal_mean_7d | 4.460     | 5.079        | -0.619 | -0.685  | -0.545  | 12.185    |

## XGBoost RB: other point metrics

| horizon | rmse  | wape   | bias   |
| ------- | ----- | ------ | ------ |
| 1       | 5.888 | 26.995 | -0.989 |
| 2       | 6.295 | 29.043 | -1.020 |
| 3       | 6.536 | 30.157 | -0.977 |
| 4       | 6.692 | 30.855 | -0.956 |
| 5       | 6.840 | 31.455 | -0.913 |
| 6       | 6.920 | 31.805 | -0.889 |

## DL data volume: WAPE (%), mean over test folds

| horizon | xgb    | persistence | seasonal_naive_1d | seasonal_naive_7d | seasonal_mean_7d |
| ------- | ------ | ----------- | ----------------- | ----------------- | ---------------- |
| 1       | 29.815 | 38.081      | 45.537            | 47.826            | 38.129           |
| 2       | 31.985 | 46.953      | 45.549            | 47.835            | 38.141           |
| 3       | 33.011 | 54.945      | 45.546            | 47.860            | 38.156           |
| 4       | 33.764 | 62.368      | 45.547            | 47.863            | 38.154           |
| 5       | 34.222 | 69.054      | 45.549            | 47.858            | 38.145           |
| 6       | 34.492 | 74.276      | 45.552            | 47.866            | 38.142           |

## Upper-bound coverage on test blocks (RB, mean over horizons and folds)

| nominal | coverage_raw | coverage_conformal |
| ------- | ------------ | ------------------ |
| 0.800   | 0.795        | 0.800              |
| 0.900   | 0.904        | 0.898              |
| 0.950   | 0.961        | 0.949              |
| 0.990   | 0.997        | 0.989              |

## Sleep-mode policies, 1 h ahead (theta = 10.0 % RB, residual power 30%)

| policy              | sleep_share | false_sleep_rate | severe_rate | displaced_share | energy_saved_pct | capture | precision | recall |
| ------------------- | ----------- | ---------------- | ----------- | --------------- | ---------------- | ------- | --------- | ------ |
| oracle              | 0.490       | 0.000            | 0.000       | 0.155           | 29.491           | 1.000   | 1.000     | 1.000  |
| persistence         | 0.491       | 0.175            | 0.004       | 0.203           | 23.673           | 0.803   | 0.825     | 0.825  |
| seasonal_naive_1d   | 0.490       | 0.202            | 0.008       | 0.220           | 22.767           | 0.772   | 0.798     | 0.799  |
| seasonal_mean_7d    | 0.440       | 0.138            | 0.003       | 0.166           | 21.752           | 0.738   | 0.862     | 0.773  |
| xgb_median          | 0.496       | 0.139            | 0.002       | 0.191           | 25.037           | 0.849   | 0.861     | 0.871  |
| xgb_conformal_a0.2  | 0.332       | 0.048            | 0.001       | 0.084           | 17.619           | 0.597   | 0.952     | 0.644  |
| xgb_conformal_a0.1  | 0.263       | 0.025            | 0.001       | 0.050           | 13.826           | 0.469   | 0.975     | 0.522  |
| xgb_conformal_a0.05 | 0.180       | 0.009            | 0.000       | 0.021           | 9.114            | 0.309   | 0.991     | 0.365  |
| xgb_conformal_a0.01 | 0.000       | -                | -           | 0.000           | 0.000            | 0.000   | -         | 0.000  |

## Sleep-mode policies, 3 h ahead (theta = 10.0 % RB, residual power 30%)

| policy              | sleep_share | false_sleep_rate | severe_rate | displaced_share | energy_saved_pct | capture | precision | recall |
| ------------------- | ----------- | ---------------- | ----------- | --------------- | ---------------- | ------- | --------- | ------ |
| oracle              | 0.491       | 0.000            | 0.000       | 0.155           | 29.523           | 1.000   | 1.000     | 1.000  |
| persistence         | 0.490       | 0.270            | 0.017       | 0.261           | 20.781           | 0.704   | 0.730     | 0.729  |
| seasonal_naive_1d   | 0.491       | 0.201            | 0.008       | 0.221           | 22.798           | 0.772   | 0.799     | 0.799  |
| seasonal_mean_7d    | 0.440       | 0.137            | 0.003       | 0.166           | 21.788           | 0.738   | 0.863     | 0.774  |
| xgb_median          | 0.488       | 0.148            | 0.003       | 0.193           | 24.255           | 0.822   | 0.852     | 0.848  |
| xgb_conformal_a0.2  | 0.319       | 0.054            | 0.001       | 0.081           | 16.699           | 0.566   | 0.946     | 0.615  |
| xgb_conformal_a0.1  | 0.243       | 0.027            | 0.000       | 0.046           | 12.597           | 0.427   | 0.973     | 0.482  |
| xgb_conformal_a0.05 | 0.144       | 0.010            | 0.000       | 0.015           | 6.999            | 0.237   | 0.990     | 0.290  |
| xgb_conformal_a0.01 | 0.000       | -                | -           | 0.000           | 0.000            | 0.000   | -         | 0.000  |

## Sleep-mode policies, 6 h ahead (theta = 10.0 % RB, residual power 30%)

| policy              | sleep_share | false_sleep_rate | severe_rate | displaced_share | energy_saved_pct | capture | precision | recall |
| ------------------- | ----------- | ---------------- | ----------- | --------------- | ---------------- | ------- | --------- | ------ |
| oracle              | 0.491       | 0.000            | 0.000       | 0.155           | 29.519           | 1.000   | 1.000     | 1.000  |
| persistence         | 0.490       | 0.401            | 0.048       | 0.357           | 16.940           | 0.574   | 0.599     | 0.598  |
| seasonal_naive_1d   | 0.491       | 0.201            | 0.008       | 0.221           | 22.789           | 0.772   | 0.799     | 0.799  |
| seasonal_mean_7d    | 0.440       | 0.137            | 0.003       | 0.166           | 21.782           | 0.738   | 0.863     | 0.774  |
| xgb_median          | 0.487       | 0.151            | 0.003       | 0.194           | 24.096           | 0.816   | 0.849     | 0.843  |
| xgb_conformal_a0.2  | 0.310       | 0.053            | 0.001       | 0.078           | 16.209           | 0.549   | 0.947     | 0.599  |
| xgb_conformal_a0.1  | 0.237       | 0.026            | 0.001       | 0.043           | 12.200           | 0.413   | 0.974     | 0.469  |
| xgb_conformal_a0.05 | 0.135       | 0.011            | 0.000       | 0.014           | 6.520            | 0.221   | 0.989     | 0.273  |
| xgb_conformal_a0.01 | 0.000       | -                | -           | 0.000           | 0.000            | 0.000   | -         | 0.000  |
