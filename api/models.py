from sqlalchemy import Column, Integer, Float, DateTime
from datetime import datetime
from api.db import Base


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow)

    usd_ngn = Column(Float, nullable=False)
    eur_ngn = Column(Float, nullable=False)
    gbp_ngn = Column(Float, nullable=False)

    prediction = Column(Float, nullable=False)
