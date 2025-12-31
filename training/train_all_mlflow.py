import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATA_PATH = "data/processed/usdngn_clean.csv"
EXPERIMENT_NAME = "fx_forecasting"


# ---------------------------
# Data preparation
# ---------------------------
def load_and_prepare_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    df["target"] = df["USD_NGN"].shift(-1)
    df = df.dropna()

    df = df.drop(columns=["Date", "USD_Rate_Category"])
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

        # 🔒 Sklearn API → must be logged with sklearn flavor
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
def evaluate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rmse_scores = []
    mae_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))

    return {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores))
    }


# ---------------------------
# Main MLflow pipeline
# ---------------------------
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_and_prepare_data(DATA_PATH)
    models = get_models()

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # Log parameters
            mlflow.log_params(model.get_params())

            # Train + evaluate
            metrics = evaluate_model(model, X, y)
            mlflow.log_metrics(metrics)

            # Retrain on full dataset
            model.fit(X, y)

            # ✅ ONE SAFE WAY FOR ALL MODELS
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            # Log feature importance (XGBoost only)
            if name == "XGBoost":
                fi_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)

                fi_path = "xgboost_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path)

            print(f"{name} logged to MLflow successfully")


if __name__ == "__main__":
    main()
