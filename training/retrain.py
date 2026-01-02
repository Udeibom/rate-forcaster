import json
import pandas as pd
from datetime import datetime, timezone

from monitoring.drift_detector import drift_exceeded
from training.train_all import train_all_models


METADATA_PATH = "data/metadata.json"
DATA_PATH = "data/processed/usdngn_clean.csv"


def new_data_available():
    # 🔧 FIX: Date column is "Date", not "date"
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    latest_data_date = df["Date"].max()

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    last_data_date = pd.to_datetime(metadata["last_data_date"])

    return latest_data_date > last_data_date, latest_data_date


def retrain_if_needed():
    new_data, latest_date = new_data_available()
    drift = drift_exceeded()

    if not new_data and not drift:
        print("✅ No retraining needed")
        return

    print("🚀 Retraining triggered")
    print(f"New data: {new_data}, Drift: {drift}")

    # 🔁 Retrain models
    metrics = train_all_models()

    # ----------------------------
    # Update metadata
    # ----------------------------
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    metadata["last_train_date"] = datetime.now(timezone.utc).isoformat()
    metadata["last_data_date"] = str(latest_date.date())
    metadata["last_rmse"] = metrics["rmse"]

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Retraining complete")


if __name__ == "__main__":
    retrain_if_needed()
