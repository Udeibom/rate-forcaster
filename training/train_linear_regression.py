import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_PATH = "data/processed/usdngn_clean.csv"
MODEL_DIR = "models"
METRICS_DIR = "evaluation"


def load_and_prepare_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    # Target: next-day USD/NGN
    df["target"] = df["USD_NGN"].shift(-1)

    # Drop last row (no future target)
    df = df.dropna()

    # Drop columns that cause leakage or are identifiers
    df = df.drop(columns=[
        "Date",
        "USD_Rate_Category"  # derived from USD_NGN
    ])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["Month", "Weekday"], drop_first=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def train_and_evaluate(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores = []
    mae_scores = []

    final_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)  # correct for new sklearn
        mae = mean_absolute_error(y_val, preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"Fold {fold} — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        final_model = model  # last fold model

    metrics = {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
    }

    return final_model, metrics


def save_artifacts(model, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/linear_regression.pkl")

    with open(f"{METRICS_DIR}/linear_regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    X, y = load_and_prepare_data(DATA_PATH)
    model, metrics = train_and_evaluate(X, y)
    save_artifacts(model, metrics)

    print("\nFinal Cross-Validated Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
