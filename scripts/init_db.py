from api.db import engine, Base
from api.models import PredictionLog

Base.metadata.create_all(bind=engine)
print("✅ Database initialized")
