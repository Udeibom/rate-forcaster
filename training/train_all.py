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

from features.pipeline import FeaturePipeline
from features.build_features import load_processed_data

from evaluation.metrics import evaluate_regression
from evaluation.dashboard import plot_actual_vs_predicted, plot_error_over_time
from evaluation.backtesting import compare_models
from evaluation.drift import rolling_error_detector, ks_drift_test
from evaluation.report import generate_monitoring_report

DATA_PATH = "data/processed/usdngn_clean.csv"
MODEL_DIR = "models"
ARTIFACTS_DIR = "artifacts/evaluation"
FEATURE_COLUMNS_PATH = f"{MODEL_DIR}/feature_columns.json"


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rmses, maes = [], []
    last_val = None

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        maes.append(mean_absolute_error(y_val, preds))
        last_val = (X_val, y_val, preds)

    return model, {
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
    }, last_val


def train_all_models():
    """
    Train all models, save artifacts, return best model metrics for programmatic retraining.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # -----------------------------
    # Load data + feature pipeline
    # -----------------------------
    raw_df = load_processed_data(DATA_PATH)

    pipeline = FeaturePipeline()
    X, y = pipeline.fit_transform(raw_df)
    joblib.dump(pipeline, f"{MODEL_DIR}/feature_pipeline.pkl")

    # -----------------------------
    # Train models
    # -----------------------------
    models = get_models()
    results, trained, val_sets = {}, {}, {}

    print("\nTraining models...\n")
    for name, model in models.items():
        trained_model, metrics, val_data = evaluate_model(model, X, y)
        results[name] = metrics
        trained[name] = trained_model
        val_sets[name] = val_data
        print(f"{name} RMSE: {metrics['rmse_mean']:.4f}")

    # -----------------------------
    # Select best model
    # -----------------------------
    best_name = min(results, key=lambda k: results[k]["rmse_mean"])
    best_model = trained[best_name]

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=4)

    # -----------------------------
    # Final evaluation
    # -----------------------------
    X_val, y_val, y_pred = val_sets[best_name]
    final_metrics = evaluate_regression(y_val, y_pred)

    print("\nFinal Evaluation")
    print(final_metrics)

    plot_actual_vs_predicted(y_val, y_pred, ARTIFACTS_DIR)
    plot_error_over_time(y_val, y_pred, ARTIFACTS_DIR)

    # -----------------------------
    # Drift detection
    # -----------------------------
    drift_df = rolling_error_detector(y_val, y_pred)
    drift_df.to_csv(f"{ARTIFACTS_DIR}/drift_flags.csv", index=False)

    errors = y_val.values - y_pred
    split = int(len(errors) * 0.7)
    ks_result = ks_drift_test(errors[:split], errors[split:])
    with open(f"{ARTIFACTS_DIR}/ks_drift_test.json", "w") as f:
        json.dump(ks_result, f, indent=4)

    # -----------------------------
    # Backtesting + report
    # -----------------------------
    backtest = compare_models(y_val, y_pred)
    pd.DataFrame([backtest]).to_csv(
        f"{ARTIFACTS_DIR}/backtest_comparison.csv", index=False
    )

    generate_monitoring_report(
        metrics=final_metrics,
        backtest_csv=f"{ARTIFACTS_DIR}/backtest_comparison.csv",
        drift_csv=f"{ARTIFACTS_DIR}/drift_flags.csv",
        ks_result=ks_result,
        output_path=f"{ARTIFACTS_DIR}/monitoring_report.md",
    )

    print(f"\n✅ Best model: {best_name}")
    print(f"📦 Feature columns saved to {FEATURE_COLUMNS_PATH}")

    # -----------------------------
    # Return minimal metrics for retraining logic
    # -----------------------------
    return {
        "rmse": float(final_metrics["rmse"]),
        "mae": float(final_metrics["mae"]),
    }


def main():
    """
    CLI entrypoint: runs full training pipeline and prints final metrics.
    """
    metrics = train_all_models()
    print("\n✅ Training complete. Best model metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
