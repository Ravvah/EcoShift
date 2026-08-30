from datetime import datetime
from typing import List
from pydantic import BaseModel

class ForecastDataPoint(BaseModel):
    timestamp: datetime
    predicted_price_eur_mwh: float
    q10_price_eur_mwh: float
    q90_price_eur_mwh: float


class PredictResponse(BaseModel):
    model_version: str
    generated_at: datetime
    predictions: List[ForecastDataPoint]
