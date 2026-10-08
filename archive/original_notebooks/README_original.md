# LTE Traffic Forecasting for Proactive RAN Optimization

Predicting cellular data volume and radio resource (PRB) utilization 1 to 6 hours ahead using machine learning on live commercial Radio Access Network (RAN) performance counters.

## Dataset & Provenance

- **Source:** Open-source Zenodo dataset (Released 2026) capturing live commercial base stations in Slovakia.
- **Citation:** Dataset 10.5281/zenodo.17815388 (Available under CC BY 4.0).
- **Scope:** 15-minute resolution PM counters for LTE 1800 MHz across 184 unique sectors.
- **Timeline:** ~3 months of data. Training is performed on the first 80%, with a strict 6-hour gap before evaluating on the final 20% to prevent target leakage.

## Technical Architecture

This project processes raw PM counters into a multi-horizon forecasting engine:

1. **Data Pipeline:** Safely aggregates multiple carriers per sector to a strict 15-minute grid. Missing PM counters are preserved as true network gaps rather than zero-filled, preventing the fabrication of traffic data.
2. **Feature Engineering:** Extracts temporal cyclical features (hour, day of week), network quality metrics (CQI, Active Users), and autoregressive lags (t-1, t-24h).
3. **Multi-Output XGBoost:** Utilizes `MultiOutputRegressor` wrapped around `XGBRegressor` (Hist Gradient Boosting) to simultaneously predict 12 targets per sector (1-6h Data Volume in MB, 1-6h RB Utilization %).

## Key Results & Evaluation (1-Hour Horizon)

To ensure the ML model adds true operational value, it is benchmarked against two standard baselines:

- **Persistence:** Forecasting that the next hour will equal the current hour.
- **Seasonal Naive:** Forecasting that the next hour will equal the same hour yesterday.

| Metric                  | XGBoost Model | Persistence Baseline | Seasonal Naive Baseline |
| :---------------------- | :------------ | :------------------- | :---------------------- |
| **RB Utilization MAE**  | **4.91%**     | 5.59%                | 7.45%                   |
| **RB Utilization R²**   | **0.73**      | N/A                  | N/A                     |
| **RB Utilization WAPE** | **30.38%**    | N/A                  | N/A                     |

## Business Impact: Zero-Touch Energy Savings

The primary operational use-case for this model is proactive base station sleep modes. If a sector is forecasted to drop below 10% RB utilization, the MNO can safely disable carriers to reduce OPEX.

We evaluated the model's ability to trigger this business rule safely:

- **Precision (85%):** When the model triggers a sleep mode, the sector genuinely experiences <10% utilization 85% of the time, protecting user SLAs.
- **Recall (73%):** The model successfully identifies and captures 73% of all possible energy-saving opportunities in the network.

## Limitations & Future Work

- **Limited History:** The dataset spans approximately 3 months. A full year of data is required to effectively learn macro-seasonal trends (holidays, summer vs. winter usage).
- **Missing Congestion Data:** The test set lacked instances of severe congestion (>85% PRB utilization), preventing the evaluation of precision/recall for proactive traffic steering algorithms.

## How to Run

1. Clone the repository and install requirements via `pip install -r requirements.txt`.
2. Download `Dataset_02_LTE_1800.csv` from [Zenodo](https://doi.org/10.5281/zenodo.17815388) and place it in the `data/` directory.
3. Execute `02_feature_engineering_all.ipynb` to build the feature matrix.
4. Execute `07_strict_evaluation.ipynb` to train the MultiOutput model and generate baseline comparisons.
