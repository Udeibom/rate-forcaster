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
