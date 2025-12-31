import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

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
    df = df.dropna()

    # Drop leakage / identifiers
    df = df.drop(columns=[
        "Date",
        "USD_Rate_Category"
    ])

    # Encode categoricals
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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        evals = [(dtrain, "train"), (dval, "val")]

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )



        preds = model.predict(xgb.DMatrix(X_val))

        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"Fold {fold} — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        final_model = model

    metrics = {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "best_iteration": int(final_model.best_iteration + 1)
    }

    return final_model, metrics

def save_feature_importance(model):
    raw_importance = model.get_score(importance_type="gain")

    fi_df = (
        pd.DataFrame(
            raw_importance.items(),
            columns=["feature", "importance"]
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs("evaluation", exist_ok=True)
    fi_df.to_csv("evaluation/xgboost_feature_importance.csv", index=False)



def save_artifacts(model, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/xgboost.pkl")

    with open(f"{METRICS_DIR}/xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    X, y = load_and_prepare_data(DATA_PATH)
    model, metrics = train_and_evaluate(X, y)

    save_artifacts(model, metrics)
    save_feature_importance(model)

    print("\nFinal Cross-Validated Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
