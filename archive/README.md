# Archive: first iteration (notebooks 01-07)

Kept for provenance. **Superseded by `src/lte_forecast/` - do not quote the numbers in `README_original.md`**
(RB MAE 4.91 %, sleep precision 85 % / recall 73 %). The direction was right (real RAN counters, chronological split,
persistence baseline, a decision-level metric); the following issues were found while productionising it, and the
new pipeline addresses each one:

| # | Issue in the first iteration | Why it matters | Fix in the new pipeline |
|---|---|---|---|
| 1 | `engineered_features_all.csv` is built with `df.dropna()`. The raw rows shown in notebook 01 have RB utilisation = NaN exactly when DL volume and active users are 0. The raw file has 1.80 M rows; the engineered table has 0.51 M | Idle slots - the sleep-mode candidates - are silently removed from training **and** test, so MAE/WAPE and the sleep precision/recall are measured on a non-representative subset | Idle slots are kept as 0 % (`data.clean`); true gaps stay NaN and only reduce the number of valid origins; `data_audit.json` reports every count |
| 2 | Seasonal-naive baseline = `shift(96)` of the *origin* value, compared with the target `t+1h` | The baseline is 25 h old instead of 24 h old, which handicaps it (reported 7.45 vs persistence 5.59) | Baselines are aligned to the target slot (`t+4h-96`); a unit test pins this (`test_seasonal_naive_is_aligned_to_target_slot`); extra baselines: 7-day-lag and 7-day seasonal mean |
| 3 | Notebooks 05/06 select `minute`, `is_weekend`, `vol_lag_4`, `vol_lag_96`, which the final `02_feature_engineering_all` no longer produces (row counts also differ: 688 k vs 411 k training rows) | The notebooks cannot be re-run in order; results are not reproducible from the committed code | One config-driven pipeline, deterministic seed, unit + smoke tests, CI |
| 4 | Final model sees only the current value, 3 other current counters and `*_lag_1` | No seasonal-at-target-time features, no rolling statistics, no per-cell context | 50+ leakage-tested features incl. same-slot yesterday / last week, rolling statistics, train-window cell statistics |
| 5 | `resample(...).agg('sum')` on volume | pandas returns 0 for an all-NaN bin, so volume gaps become zeros while RB stays NaN | Panel built with explicit NaN for missing slots |
| 6 | Volume unit labelled MB in the README and GB in the plots; errors of 200-400 "GB" | Unverifiable claim | Unit-agnostic code, relative metrics (WAPE) for volume |
| 7 | Sleep-mode precision/recall only for the model; no policy baseline | A reader cannot tell whether 85 %/73 % is good | Persistence / seasonal / oracle policies, false-sleep and displaced-traffic metrics, conformal upper-bound policy, threshold-sweep frontier |
| 8 | Single 80/20 split | One test window can be lucky | 3 rolling-origin folds + block-bootstrap CIs |

Issues 1-3 mean the first-iteration numbers are likely to change once the pipeline is run on the full file. That is expected.
