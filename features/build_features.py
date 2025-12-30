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


# -----------------------
# Baseline Time-Series Features
# -----------------------

def create_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Lag features capture momentum and mean reversion effects.
    They represent past observed exchange rates available at prediction time.
    """
    for lag in lags:
        df[f"lag_{lag}"] = df["USD_NGN"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Rolling statistics summarize recent historical behavior.
    We shift by 1 to avoid using information from the current timestep.
    """
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df["USD_NGN"].shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["USD_NGN"].shift(1).rolling(window).std()
        )
    return df


# -----------------------
# Calendar Features
# -----------------------

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features capture seasonal and weekly patterns
    observed in financial time series.
    """
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    return df


# -----------------------
# Technical Indicators
# -----------------------

def create_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns measure relative price changes and capture momentum.
    """
    df["returns"] = df["USD_NGN"].pct_change()
    return df


def create_volatility_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Volatility proxies measure market uncertainty.
    Rolling standard deviation of returns is a common FX risk metric.
    """
    for window in windows:
        df[f"rolling_volatility_{window}"] = (
            df["returns"].shift(1).rolling(window).std()
        )
    return df


# -----------------------
# Target Definition
# -----------------------

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict next-day exchange rate (t+1).
    """
    df["target"] = df["USD_NGN"].shift(-1)
    return df


def build_features() -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df = load_processed_data(PROCESSED_DATA_PATH)

    # Baseline time-series features
    df = create_lag_features(df, lags=[1, 7, 14, 30])
    df = create_rolling_features(df, windows=[7, 14, 30])

    # Calendar features
    df = create_calendar_features(df)

    # Technical indicators
    df = create_return_features(df)
    df = create_volatility_features(df, windows=[7, 14, 30])

    # Target
    df = create_target(df)

    # Remove rows with missing values from shifting/rolling
    df = df.dropna().reset_index(drop=True)

    return df


def save_features(df: pd.DataFrame, path: Path):
    """
    Save final feature dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print("Building calendar and technical features...")

    feature_df = build_features()
    save_features(feature_df, FEATURES_PATH)

    print(f"Feature dataset saved to {FEATURES_PATH}")
