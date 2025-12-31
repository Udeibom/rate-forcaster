import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURE_MATRIX_PATH = Path("features/final_feature_matrix.csv")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Load engineered feature matrix.
    """
    return pd.read_csv(path, parse_dates=["Date"])


def evaluate_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
):
    """
    Perform expanding-window time-series cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)

        print(
            f"Fold {fold} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%"
        )

    return {
        "RMSE_mean": np.mean(rmse_scores),
        "MAE_mean": np.mean(mae_scores),
        "MAPE_mean": np.mean(mape_scores),
    }


if __name__ == "__main__":
    print("Running time-series cross-validation...")

    df = load_feature_matrix(FEATURE_MATRIX_PATH)

    # Use numeric features only (avoids categorical leakage)
    X = df.select_dtypes(include=["number"]).drop(columns=["target"])
    y = df["target"]

    metrics = evaluate_model_cv(X, y)

    print("\nAverage CV performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
