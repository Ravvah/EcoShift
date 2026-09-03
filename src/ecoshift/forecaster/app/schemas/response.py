from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class ForecastDataPoint(BaseModel):
    timestamp: datetime
    predicted_price_eur_mwh: float
    predicted_co2_g_kwh: Optional[float]
    q10_price_eur_mwh: Optional[float]
    q90_price_eur_mwh: Optional[float]


class PredictResponse(BaseModel):
    model_version: str
    generated_at: datetime
    predictions: List[ForecastDataPoint]
