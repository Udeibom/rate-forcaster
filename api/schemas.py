from pydantic import BaseModel
from datetime import date


class PredictionRequest(BaseModel):
    Date: date
    USD_NGN: float


class PredictionResponse(BaseModel):
    prediction: float
