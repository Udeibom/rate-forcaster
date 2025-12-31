import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_PATH = "data/processed/usdngn_clean.csv"
MODEL_DIR = "models"
METRICS_DIR = "evaluation"


# ---------------------------
# Data preparation
# ---------------------------
def load_and_prepare_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    df["target"] = df["USD_NGN"].shift(-1)
    df = df.dropna()

    df = df.drop(columns=[
        "Date",
        "USD_Rate_Category"
    ])

    df = pd.get_dummies(df, columns=["Month", "Weekday"], drop_first=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


# ---------------------------
# Model factory
# ---------------------------
def get_models():
    return {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }


# ---------------------------
# Training & evaluation
# ---------------------------
def evaluate_model(name, model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    metrics = {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores))
    }

    return model, metrics


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X, y = load_and_prepare_data(DATA_PATH)
    models = get_models()

    results = {}
    trained_models = {}

    print("\nTraining models...\n")

    for name, model in models.items():
        trained_model, metrics = evaluate_model(name, model, X, y)
        results[name] = metrics
        trained_models[name] = trained_model

        with open(f"{METRICS_DIR}/{name.lower()}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"{name} RMSE: {metrics['rmse_mean']:.4f}")

    # ---------------------------
    # Model selection
    # ---------------------------
    best_model_name = min(results, key=lambda k: results[k]["rmse_mean"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    with open(f"{METRICS_DIR}/best_model.json", "w") as f:
        json.dump(
            {
                "model_name": best_model_name,
                "metrics": best_metrics
            },
            f,
            indent=4
        )

    print("\nBest model selected:")
    print(f"Model: {best_model_name}")
    print(f"RMSE: {best_metrics['rmse_mean']:.4f}")


if __name__ == "__main__":
    main()
