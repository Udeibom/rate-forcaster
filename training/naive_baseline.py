import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURE_MATRIX_PATH = Path("features/final_feature_matrix.csv")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Load engineered feature matrix.
    """
    return pd.read_csv(path, parse_dates=["Date"])


def naive_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Naive forecast: y(t+1) = y(t)
    """
    return df["USD_NGN"].values


def backtest_naive_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    split_idx = int(len(df) * (1 - test_size))
    test_df = df.iloc[split_idx:]

    y_true = test_df["target"].values[1:]
    y_pred = naive_predict(test_df)[1:]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }



if __name__ == "__main__":
    print("Running naive baseline backtest...")

    df = load_feature_matrix(FEATURE_MATRIX_PATH)
    metrics = backtest_naive_model(df)

    print("Naive baseline performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
