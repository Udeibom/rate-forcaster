import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/usdngn_raw.csv")
PROCESSED_DATA_PATH = Path("data/processed/usdngn_clean.csv")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw USD/NGN exchange rate data.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess exchange rate data:
    - Sort by date
    - Handle missing dates
    - Handle missing values
    """
    # Sort by time
    df = df.sort_values("date").reset_index(drop=True)

    # Set date as index for time-series operations
    df = df.set_index("date")

    # Create continuous daily date range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    # Rename index back to 'date'
    df.index.name = "date"

    # Handle missing values (forward fill is standard for FX rates)
    df = df.ffill()

    return df.reset_index()


def save_processed_data(df: pd.DataFrame, path: Path):
    """
    Save cleaned dataset to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print("Starting data ingestion pipeline...")

    raw_df = load_data(RAW_DATA_PATH)
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df, PROCESSED_DATA_PATH)

    print(f"Data ingestion complete. Saved to {PROCESSED_DATA_PATH}")
