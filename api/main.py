from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import json

# ----------------------------
# Resolve paths safely
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PIPELINE_PATH = BASE_DIR / "models" / "feature_pipeline.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "usdngn_clean.csv"

# ----------------------------
# Load artifacts
# ----------------------------
model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="USD/NGN FX Forecast API")


class PredictionRequest(BaseModel):
    Date: str
    USD_NGN: float
    EUR_NGN: float
    GBP_NGN: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # ----------------------------
        # Load historical raw data
        # ----------------------------
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

        # ----------------------------
        # Append new row
        # ----------------------------
        new_row = {
            "Date": pd.to_datetime(req.Date),
            "USD_NGN": req.USD_NGN,
            "EUR_NGN": req.EUR_NGN,
            "GBP_NGN": req.GBP_NGN,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # ----------------------------
        # Feature engineering
        # ----------------------------
        X = pipeline.transform(df)

        X_latest = X.iloc[[-1]][FEATURE_COLUMNS]

        # ----------------------------
        # Predict
        # ----------------------------
        prediction = model.predict(X_latest)[0]

        return {
            "prediction": float(prediction),
            "date": req.Date,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
