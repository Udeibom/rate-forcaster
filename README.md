# End-to-End Exchange Rate Forecasting System (USD/NGN)

## Problem Statement

Exchange rate volatility significantly impacts businesses, investors, and individuals in emerging economies such as Nigeria. Reliable short-term forecasts of the USD/NGN exchange rate can support better financial planning and risk management. However, most forecasting approaches lack robustness, monitoring, and production readiness.

This project aims to build a production-ready machine learning system that forecasts the USD/NGN exchange rate using historical time-series data, with automated model selection, performance tracking, and live inference capabilities.

## Forecasting Objective

- **Target Variable:** USD/NGN closing exchange rate
- **Forecast Horizons:**
  - 1-day ahead (primary)
  - 7-day ahead (secondary)
- **Prediction Type:** Regression (continuous numeric value)

## Evaluation Metrics

Models will be evaluated using:
- **RMSE (Root Mean Squared Error)** – primary metric
- **MAE (Mean Absolute Error)** – interpretability
- **MAPE (Mean Absolute Percentage Error)** – relative error

## Constraints & Assumptions

- Only historical exchange rate data is used initially
- Time-order must be preserved (no random shuffling)
- System must support retraining and monitoring

---

## Project Structure

```text
fx-forecasting-ml/
├── data/
│   ├── raw/                 # Raw exchange rate data
│   └── processed/           # Cleaned and versioned datasets
├── features/
│   ├── build_features.py    # Feature engineering logic
│   └── pipeline.py          # Reusable feature pipeline
├── training/
│   ├── train_all_mlflow.py  # Unified training + experiment tracking
│   └── register_best_model.py # Model registry & promotion
├── evaluation/
│   ├── *_metrics.json       # Stored evaluation metrics
│   └── xgboost_feature_importance.csv
├── models/
│   └── *.pkl                # Trained model artifacts
├── mlflow/                  # MLflow experiment tracking data
└── README.md

## Training Pipeline

The model training pipeline is fully script-based and reproducible.

### Steps

1. **Feature Preparation**
   - Lag features, rolling statistics, and calendar effects
   - No future data leakage (time-aware splits)

2. **Model Training**
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost with early stopping

3. **Evaluation**
   - Expanding-window time-series cross-validation
   - Metrics: RMSE, MAE

4. **Experiment Tracking**
   - MLflow logs parameters, metrics, and artifacts
   - Each model is tracked as a separate run

5. **Model Selection**
   - Best model selected automatically based on RMSE

6. **Model Registry**
   - Best-performing model registered in MLflow
   - Promoted to Production stage with versioning

### Reproducibility

All training steps can be reproduced by running:

```bash
python training/train_all_mlflow.py
python training/register_best_model.py
