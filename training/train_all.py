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

from evaluation.metrics import (
    evaluate_regression,
    plot_residuals,
    plot_predictions
)

from evaluation.dashboard import (
    plot_actual_vs_predicted,
    plot_error_over_time
)

from evaluation.backtesting import compare_models
from evaluation.drift import rolling_error_detector, ks_drift_test
from evaluation.report import generate_monitoring_report


# ---------------------------
# Paths
# ---------------------------
DATA_PATH = "data/processed/usdngn_clean.csv"
MODEL_DIR = "models"
METRICS_DIR = "evaluation"
ARTIFACTS_DIR = "artifacts/evaluation"


# ---------------------------
# Data preparation
# ---------------------------
def load_and_prepare_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    # Target: next-day USD/NGN
    df["target"] = df["USD_NGN"].shift(-1)
    df = df.dropna()

    # Drop leakage / identifiers
    df = df.drop(columns=["Date", "USD_Rate_Category"])

    # Encode categoricals
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
# Cross-validation evaluation
# ---------------------------
def evaluate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores = []
    mae_scores = []
    last_val_data = None

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))

        last_val_data = (X_val, y_val, preds)

    metrics = {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
    }

    return model, metrics, last_val_data


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y = load_and_prepare_data(DATA_PATH)
    models = get_models()

    results = {}
    trained_models = {}
    val_sets = {}

    print("\nTraining models...\n")

    for name, model in models.items():
        trained_model, metrics, val_data = evaluate_model(model, X, y)

        results[name] = metrics
        trained_models[name] = trained_model
        val_sets[name] = val_data

        with open(f"{METRICS_DIR}/{name.lower()}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"{name} RMSE: {metrics['rmse_mean']:.4f}")

    # ---------------------------
    # Select best model
    # ---------------------------
    best_model_name = min(results, key=lambda k: results[k]["rmse_mean"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    with open(f"{METRICS_DIR}/best_model.json", "w") as f:
        json.dump(
            {"model_name": best_model_name, "metrics": best_metrics},
            f,
            indent=4
        )

    # ---------------------------
    # Final evaluation
    # ---------------------------
    X_val, y_val, y_pred = val_sets[best_model_name]

    final_metrics = evaluate_regression(y_val, y_pred)

    print("\nFinal Evaluation (Best Model)")
    print(f"RMSE: {final_metrics['rmse']:.4f}")
    print(f"MAE: {final_metrics['mae']:.4f}")
    print(f"MAPE: {final_metrics['mape']:.2f}%")

    # Core plots
    plot_residuals(y_val, y_pred, ARTIFACTS_DIR)
    plot_predictions(y_val, y_pred, ARTIFACTS_DIR)

    # Dashboard plots
    plot_actual_vs_predicted(y_val, y_pred, ARTIFACTS_DIR)
    plot_error_over_time(y_val, y_pred, ARTIFACTS_DIR)

    # ---------------------------
    # Drift detection
    # ---------------------------
    print("\nRunning drift detection...")

    drift_df = rolling_error_detector(y_val, y_pred)
    drift_df.to_csv(f"{ARTIFACTS_DIR}/drift_flags.csv", index=False)

    errors = y_val.values - y_pred
    split = int(len(errors) * 0.7)

    ks_result = ks_drift_test(errors[:split], errors[split:])

    with open(f"{ARTIFACTS_DIR}/ks_drift_test.json", "w") as f:
        json.dump(ks_result, f, indent=4)

    print("\nKS Drift Test:")
    print(f"KS Statistic: {ks_result['ks_statistic']:.4f}")
    print(f"P-value: {ks_result['p_value']:.6f}")
    print(f"Drift detected: {ks_result['drift_detected']}")

    # ---------------------------
    # Backtesting
    # ---------------------------
    backtest_results = compare_models(y_val, y_pred)

    backtest_df = pd.DataFrame([backtest_results])
    backtest_df.to_csv(
        f"{ARTIFACTS_DIR}/backtest_comparison.csv",
        index=False
    )

    print("\nBacktesting Results (Model vs Naive)")
    print(backtest_df.T)

    # ---------------------------
    # Monitoring report
    # ---------------------------
    metrics_summary = {
        "rmse": final_metrics["rmse"],
        "mae": final_metrics["mae"],
        "mape": final_metrics["mape"],
    }

    report = generate_monitoring_report(
        metrics=metrics_summary,
        backtest_csv=f"{ARTIFACTS_DIR}/backtest_comparison.csv",
        drift_csv=f"{ARTIFACTS_DIR}/drift_flags.csv",
        ks_result=ks_result,
        output_path=f"{ARTIFACTS_DIR}/monitoring_report.md"
    )

    print("\nMonitoring Report Generated:")
    print(report)

    print("\nEvaluation artifacts saved to:", ARTIFACTS_DIR)
    print(f"Best model selected: {best_model_name}")


if __name__ == "__main__":
    main()
