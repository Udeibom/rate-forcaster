import pandas as pd
from pathlib import Path
from typing import Tuple

from features.build_features import (
    load_processed_data,
    create_lag_features,
    create_rolling_features,
    create_calendar_features,
    create_return_features,
    create_volatility_features,
    create_target,
)

PROCESSED_DATA_PATH = Path("data/processed/usdngn_clean.csv")
FEATURE_MATRIX_PATH = Path("features/final_feature_matrix.csv")


class FeaturePipeline:
    """
    Feature engineering pipeline for time-series forecasting.
    Ensures leakage-safe transformations and reproducibility.
    """

    def __init__(
        self,
        lags=[1, 7, 14, 30],
        rolling_windows=[7, 14, 30],
        volatility_windows=[7, 14, 30],
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.volatility_windows = volatility_windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.
        """
        df = create_lag_features(df, self.lags)
        df = create_rolling_features(df, self.rolling_windows)
        df = create_calendar_features(df)
        df = create_return_features(df)
        df = create_volatility_features(df, self.volatility_windows)
        df = create_target(df)

        # Drop rows affected by shifting/rolling
        df = df.dropna().reset_index(drop=True)

        return df

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Time-based train-test split (no shuffling).
        """
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]

        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        return X_train, X_test, y_train, y_test


def build_and_save_feature_matrix():
    """
    Build full feature matrix and save to disk.
    """
    df = load_processed_data(PROCESSED_DATA_PATH)

    pipeline = FeaturePipeline()
    feature_df = pipeline.transform(df)

    FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_MATRIX_PATH, index=False)

    print(f"Final feature matrix saved to {FEATURE_MATRIX_PATH}")


if __name__ == "__main__":
    build_and_save_feature_matrix()
