
# End-to-End Exchange Rate Forecasting System (USD/NGN)

## Overview

Exchange rate volatility significantly affects businesses, investors, and individuals in emerging economies such as Nigeria.  
Rather than building a single predictive model in a notebook, this project was designed as a **production-style machine learning system** that handles the **entire model lifecycle** — from feature engineering and training to monitoring, drift detection, retraining, and API-based inference.

The system forecasts the **USD/NGN exchange rate** using historical time-series data and is built to be **robust, extensible, and deployment-ready**.

---

## Problem Statement

Traditional exchange rate forecasting approaches often:
- Rely on static models
- Ignore concept drift
- Lack monitoring and retraining mechanisms
- Are not production-ready

As market dynamics evolve, these models degrade silently.

**Goal:**  
Build an end-to-end ML system that can:
- Train and compare multiple models
- Automatically select the best model
- Monitor performance over time
- Detect drift and retrain when needed
- Serve predictions via an API

---

## Forecasting Objective

- **Target Variable:** USD/NGN closing exchange rate  
- **Prediction Type:** Regression  
- **Forecast Horizons:**
  - 1-day ahead (primary)
  - 7-day ahead (secondary)

---

## Evaluation Metrics

Models are evaluated using:

- **RMSE (Root Mean Squared Error)** — primary metric
- **MAE (Mean Absolute Error)** — interpretability
- **MAPE (Mean Absolute Percentage Error)** — relative error

Time ordering is strictly preserved (no random shuffling).

---

## System Architecture

```
                 ┌────────────────────┐
                 │   Raw FX Data       │
                 │ (CSV / External)    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Feature Pipeline   │
                 │ (shared logic)     │
                 └─────────┬──────────┘
                           │
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Linear Model   │ │ Random Forest  │ │ XGBoost        │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        └──────────────┬──────────────────────┘
                       ▼
             ┌────────────────────┐
             │ Best Model Selector│
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ Model Registry     │
             │ (Production)       │
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ FastAPI Inference  │
             │ + SQLite Logging   │
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ Monitoring & Drift │
             │ Detection          │
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ Automated Retrain  │
             └────────────────────┘
```

---

## Data Flow

```
New FX Data
   │
   ▼
Feature Pipeline
   │
   ▼
Model Training / Inference
   │
   ├── Predictions → SQLite DB
   │
   ├── Metrics → Evaluation Module
   │
   ├── Drift Detection
   │
   ▼
Retraining Trigger (if needed)
```

---

## Feature Engineering

A **shared feature pipeline** is used for both training and inference to avoid feature skew.

Features include:
- Lag features (1, 7, 14, 30 days)
- Rolling statistics
- Returns
- Volatility measures
- Calendar features

The pipeline follows a **scikit-learn compatible API** (`fit`, `transform`, `fit_transform`) and serves as the single source of truth.

---

## Models Trained

The following models are trained and compared:

- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

Time-series cross-validation is used, and the model with the lowest mean RMSE is promoted to production.

---

## Model Selection & Registry

- All models are evaluated under the same pipeline
- Best model is automatically selected
- Production model artifacts are saved:
  - Model weights
  - Feature pipeline
  - Feature column schema

This enables consistent inference and retraining.

---

## Monitoring & Drift Detection

The system continuously evaluates model health using:

- Rolling error analysis
- KS-test for distribution shift
- Backtesting against naive baselines

Drift flags and evaluation metrics are stored as artifacts and summarized in automated reports.

---

## Automated Retraining

Retraining is triggered when:
- New data arrives
- Drift thresholds are exceeded

A scheduler-ready retraining script simulates cron-like behavior locally and is designed to scale to production environments.

---

## Prediction API

Predictions are served using **FastAPI**.

### Endpoint
```
POST /predict
```

### Flow
API → Feature Pipeline → Production Model → Prediction

Predictions and timestamps are logged into a SQLite database (Postgres-ready).

---

## Project Structure

```
.
├── api/
│   └── main.py
├── data/
│   └── processed/
├── features/
│   ├── build_features.py
│   └── pipeline.py
├── training/
│   ├── train_all.py
│   ├── retrain.py
│   └── scheduler.py
├── evaluation/
│   ├── metrics.py
│   ├── dashboard.py
│   ├── drift.py
│   ├── backtesting.py
│   └── report.py
├── models/
├── artifacts/
├── README.md
└── requirements.txt
```

---

## Running the Project Locally

### 1. Create virtual environment
```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Train models
```
python training/train_all.py
```

### 4. Start API
```
uvicorn api.main:app --reload
```

### 5. Make prediction
```
POST http://127.0.0.1:8000/predict
```

---

## Interview Gold

> This project was built as a full ML system rather than a notebook experiment.
> Emphasis was placed on feature consistency, model lifecycle management,
> drift monitoring, and automated retraining — closely mirroring real-world
> production ML workflows.

---

## Future Improvements

- Dockerization & cloud deployment
- CI/CD pipelines
- Alerting (Slack / Email)
- Shadow model comparisons
- Live data ingestion

---

## Author

**Caleb Udeibom**  
Computer Science Graduate | Aspiring ML Engineer  
