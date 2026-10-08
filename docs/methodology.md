# Methodology

## 1. Question and decision framing
*Can machine learning forecast LTE sector load - DL data volume and PRB/RB utilisation - 1 to 6 hours ahead on real
operational counters, how much better than strong naive forecasters, and does the forecast support a safe
energy-saving decision?*

The decision studied is **carrier sleep mode**: put a capacity cell to sleep in slot `t+h` when the load is predicted to be
negligible (RB utilisation below `theta`, default 10 %). A wrong "sleep" hurts users; a missed "sleep" wastes energy.
A point forecast hides this asymmetry, so the pipeline also produces **calibrated upper bounds** and evaluates policies that
sleep only when the bound is below `theta`.

## 2. Data handling (`data.py`)
| Step | Rule | Rationale |
|---|---|---|
| Exact duplicate rows | dropped | repeated export lines carry no information |
| Same (cell, slot) key, different values | volumes/users **summed**, ratios/utilisation/energy **averaged** | multi-carrier rows. Utilisation is not bandwidth-weighted (bandwidth is not in the file) - a documented approximation |
| Negative counters | set to NaN | impossible values |
| RB utilisation NaN **and** DL volume = 0 **and** active DL users = 0 | set to 0 % | The counter is not computed when nothing is scheduled. Dropping these slots would remove the sleep candidates. `data_audit.json` reports the share of NaN-RB rows that satisfy this rule on the real file - verify it is ~100 % |
| CQI = 0 | NaN | CQI index 0 means "out of range / no report" |
| Slot with none of {RB, DL volume, DL users} | gap (NaN) | never zero-filled; gap slots are only excluded as origins/targets |
| Cells observed in < 50 % of the timeline | dropped | too little history to evaluate |

## 3. Leakage controls
* Features at origin `t` use slots `<= t`. `tests/test_features.py` corrupts all data after `t0` and asserts that features at origins `<= t0` are unchanged (dynamic, horizon-aligned and calendar-independent features).
* Seasonal inputs are aligned to the **target** slot (`t+4h-96`, `t+4h-672`); they are known at `t` for any horizon up to 24 h.
* Per-cell statistics (mean/p90 RB, idle share, mean volume) are computed on the training window only.
* Splits are defined on the **target slot**: training targets < calibration block < test block. The embargo of one horizon follows from this definition and is unit-tested.
* The calibration block is never used to fit or early-stop the model (fixed number of rounds, no tuning on test).

## 4. Models
One direct multi-horizon strategy: a separate gradient-boosted model per horizon and target (native XGBoost, `reg:quantileerror`, one multi-quantile booster).
RB utilisation is modelled on its natural scale; DL volume on `log1p` (quantiles are equivariant to monotone transforms, so back-transformed quantiles remain valid).
The 0.5 quantile is the point forecast (optimal for MAE); higher quantiles feed the upper bounds. Quantiles are sorted to prevent crossing.
The global model is shared by all cells; cell identity enters only through training-window statistics.

## 5. Baselines
`persistence` (current value), `seasonal_naive_1d` (same time yesterday), `seasonal_naive_7d` (same time last week) and `seasonal_mean_7d`
(mean of the same time-of-day over the last 7 days). All are aligned to the target slot. Every method is scored on the **same rows** (those
where all baselines are defined). The "strongest baseline" per horizon is chosen by lowest *test* MAE - deliberately the hardest comparison.

## 6. Evaluation protocol
* 3 rolling-origin folds, each: expanding training window | 7-day calibration block | 7-day test block (configurable).
* Metrics: MAE, RMSE, WAPE, bias per fold; pooled paired **block bootstrap** (blocks = cell-day) 95 % CI for `MAE(model) - MAE(best baseline)`; skill = `1 - MAE_model / MAE_baseline`.
* R^2 is not headlined: with strong daily seasonality it mostly measures that a daily cycle exists.

## 7. Uncertainty: one-sided split conformal (CQR style)
For level `1-alpha`, the quantile prediction is shifted by the `ceil((n+1)(1-alpha))`-th smallest calibration residual `y - q`. Guarantees assume exchangeability, which time series
violate (drift, weekly cycle), so **empirical coverage is re-measured on every test block** (`calibration.csv`, `figures/calibration_rb.png`).
Marginal coverage can still hide poor *conditional* coverage in the low-load region that matters for sleeping - that is why the policy table reports the realised false-sleep rate, not only coverage.

## 8. Sleep-mode policy analysis (`sleep.py`)
Policies: oracle; persistence; seasonal naive/mean; XGBoost median `< theta`; XGBoost conformal upper bound `< theta` for each `alpha`; and a **threshold sweep** of the median forecast (`< 1, 2, ... 10 %`).
The sweep is the fair competitor for the conformal bound: it asks whether calibrated bounds buy anything over simply choosing a more conservative threshold on a point forecast. Either result is reportable.

| Metric | Meaning |
|---|---|
| false_sleep_rate | P(actual RB >= theta given the policy slept) |
| severe_rate | same with `severe_theta` (30 %) |
| displaced_share | share of DL volume that fell in slept slots (would need to be absorbed by other layers) |
| energy_saved_pct | radio-unit energy avoided by *correct* sleeps, in % of total radio-unit energy of the test cells |
| capture | `energy_saved / energy_saved(oracle)` |

**Energy model assumptions (important).** The dataset has no sleep-state ground truth. Savings are a scenario: a sleeping radio unit is assumed to keep `sleep_power_fraction` (30 %) of its energy; a false sleep is assumed to be detected and the cell woken, so no saving is credited.
Radio-unit energy may be shared between bands/sectors, and the 1800 MHz layer can only be switched off if other layers provide coverage - neither is observable in this file. Wake-up latency and minimum dwell times are not modelled.
Treat `energy_saved_pct` as a relative figure for ranking policies, not as a forecast of operator savings.

## 9. Known limitations
* About three months of data: no yearly seasonality, few holidays; congestion (>85 % RB) is rare, so proactive-steering metrics are not evaluated (the audit prints the exact share).
* One operator, one band, one country.
* No spatial/neighbour features (cell coordinates are not published); no hyper-parameter search (fixed, moderate settings, by design, to keep the calibration block untouched).
* Conformal calibration is global, not per cell or per load regime.

## 10. Ideas for extension
Per-cell or load-bucket (Mondrian) calibration; adaptive conformal inference for drift; the 5G NR and 800/2100 MHz files for cross-layer sleep decisions; sequence models (TFT/LSTM) against the boosted-tree baseline; a dwell-time constrained sleep scheduler.
