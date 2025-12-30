import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/usdngn_clean.csv")
FEATURES_PATH = Path("features/usdngn_features.csv")


def load_processed_data(path: Path) -> pd.DataFrame:
    """
    Load processed USD/NGN dataset.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def create_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Create lag features for USD/NGN.
    """
    for lag in lags:
        df[f"lag_{lag}"] = df["USD_NGN"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Create rolling mean and std features.
    """
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df["USD_NGN"].shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["USD_NGN"].shift(1).rolling(window).std()
        )
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable: USD_NGN(t+1)
    """
    df["target"] = df["USD_NGN"].shift(-1)
    return df


def build_features() -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df = load_processed_data(PROCESSED_DATA_PATH)

    df = create_lag_features(df, lags=[1, 7, 14, 30])
    df = create_rolling_features(df, windows=[7, 14, 30])
    df = create_target(df)

    # Drop rows with NaNs caused by shifting/rolling
    df = df.dropna().reset_index(drop=True)

    return df


def save_features(df: pd.DataFrame, path: Path):
    """
    Save feature dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print("Building time-series features...")

    feature_df = build_features()
    save_features(feature_df, FEATURES_PATH)

    print(f"Feature dataset saved to {FEATURES_PATH}")
